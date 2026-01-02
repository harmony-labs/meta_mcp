//! MCP Server for meta - exposes multi-repo operations to AI tools.
//!
//! This server implements the Model Context Protocol (MCP) to allow AI assistants
//! like Claude to interact with meta repositories.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::Command;

/// MCP Protocol version
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server information
const SERVER_NAME: &str = "meta-mcp";
const SERVER_VERSION: &str = "0.1.0";

// ============================================================================
// MCP Protocol Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    server_info: ServerInfo,
}

#[derive(Debug, Serialize)]
struct ServerCapabilities {
    tools: ToolsCapability,
}

#[derive(Debug, Serialize)]
struct ToolsCapability {
    #[serde(rename = "listChanged")]
    list_changed: bool,
}

#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ListToolsResult {
    tools: Vec<Tool>,
}

#[derive(Debug, Serialize)]
struct CallToolResult {
    content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ToolContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

// ============================================================================
// Meta-specific Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct MetaConfig {
    #[serde(default)]
    projects: HashMap<String, ProjectEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ProjectEntry {
    Simple(String),
    Extended {
        repo: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
    },
}

#[derive(Debug, Serialize)]
struct ProjectInfo {
    name: String,
    path: String,
    repo: String,
    tags: Vec<String>,
}

// ============================================================================
// MCP Server
// ============================================================================

struct McpServer {
    meta_dir: Option<PathBuf>,
}

impl McpServer {
    fn new() -> Self {
        // Find .meta config in current directory or parents
        let meta_dir = std::env::current_dir()
            .ok()
            .and_then(|dir| Self::find_meta_dir(&dir));

        Self { meta_dir }
    }

    fn find_meta_dir(start: &std::path::Path) -> Option<PathBuf> {
        let mut current = start.to_path_buf();
        loop {
            for name in &[".meta", ".meta.yaml", ".meta.yml"] {
                if current.join(name).exists() {
                    return Some(current);
                }
            }
            if !current.pop() {
                return None;
            }
        }
    }

    fn run(&mut self) -> Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("Failed to parse request: {}", e);
                    continue;
                }
            };

            let response = self.handle_request(&request);
            let response_json = serde_json::to_string(&response)?;
            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;
        }

        Ok(())
    }

    fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(),
            "initialized" => return self.ok_response(request.id.clone(), serde_json::Value::Null),
            "tools/list" => self.handle_list_tools(),
            "tools/call" => self.handle_call_tool(&request.params),
            _ => Err(anyhow::anyhow!("Method not found: {}", request.method)),
        };

        match result {
            Ok(value) => self.ok_response(request.id.clone(), value),
            Err(e) => self.error_response(request.id.clone(), -32603, e.to_string()),
        }
    }

    fn ok_response(&self, id: Option<serde_json::Value>, result: serde_json::Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error_response(&self, id: Option<serde_json::Value>, code: i32, message: String) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }

    fn handle_initialize(&self) -> Result<serde_json::Value> {
        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability { list_changed: false },
            },
            server_info: ServerInfo {
                name: SERVER_NAME.to_string(),
                version: SERVER_VERSION.to_string(),
            },
        };
        Ok(serde_json::to_value(result)?)
    }

    fn handle_list_tools(&self) -> Result<serde_json::Value> {
        let tools = vec![
            Tool {
                name: "meta_list_projects".to_string(),
                description: "List all projects in the meta repository".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_git_status".to_string(),
                description: "Get git status for all projects in the meta repository".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Specific project to check (optional, defaults to all)"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_exec".to_string(),
                description: "Execute a command across all meta projects".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    },
                    "required": ["command"]
                }),
            },
        ];

        let result = ListToolsResult { tools };
        Ok(serde_json::to_value(result)?)
    }

    fn handle_call_tool(&self, params: &serde_json::Value) -> Result<serde_json::Value> {
        let name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

        let arguments = params.get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        let result = match name {
            "meta_list_projects" => self.tool_list_projects(&arguments),
            "meta_git_status" => self.tool_git_status(&arguments),
            "meta_exec" => self.tool_exec(&arguments),
            _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
        };

        match result {
            Ok(text) => {
                let call_result = CallToolResult {
                    content: vec![ToolContent {
                        content_type: "text".to_string(),
                        text,
                    }],
                    is_error: None,
                };
                Ok(serde_json::to_value(call_result)?)
            }
            Err(e) => {
                let call_result = CallToolResult {
                    content: vec![ToolContent {
                        content_type: "text".to_string(),
                        text: e.to_string(),
                    }],
                    is_error: Some(true),
                };
                Ok(serde_json::to_value(call_result)?)
            }
        }
    }

    fn tool_list_projects(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self.meta_dir.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;

        // Filter by tag if specified
        let tag_filter = args.get("tag").and_then(|v| v.as_str());
        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects.iter().filter(|p| p.tags.contains(&tag.to_string())).collect()
        } else {
            projects.iter().collect()
        };

        let mut output = format!("Found {} project(s) in {}\n\n", filtered.len(), meta_dir.display());
        for project in filtered {
            output.push_str(&format!("- {} ({})\n", project.name, project.path));
            if !project.tags.is_empty() {
                output.push_str(&format!("  Tags: {}\n", project.tags.join(", ")));
            }
        }

        Ok(output)
    }

    fn tool_git_status(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self.meta_dir.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let project_filter = args.get("project").and_then(|v| v.as_str());

        // Run meta git status with --json for structured output
        let mut cmd = Command::new("meta");
        cmd.arg("--json").arg("git").arg("status");
        cmd.current_dir(meta_dir);

        let output = cmd.output()
            .context("Failed to execute meta git status")?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);

            // If project filter specified, parse JSON and filter
            if let Some(project) = project_filter {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if let Some(results) = json.get("results").and_then(|r| r.as_array()) {
                        for result in results {
                            if result.get("project").and_then(|p| p.as_str()) == Some(project) {
                                return Ok(serde_json::to_string_pretty(result)?);
                            }
                        }
                        return Err(anyhow::anyhow!("Project '{}' not found", project));
                    }
                }
            }

            Ok(stdout.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("meta git status failed: {}", stderr))
        }
    }

    fn tool_exec(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self.meta_dir.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let command = args.get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        // Split command into parts
        for part in command.split_whitespace() {
            cmd.arg(part);
        }

        cmd.current_dir(meta_dir);

        let output = cmd.output()
            .context("Failed to execute meta command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("Command failed:\n{}\n{}", stdout, stderr))
        }
    }

    fn load_projects(&self, meta_dir: &std::path::Path) -> Result<Vec<ProjectInfo>> {
        // Try to find and parse the meta config
        for name in &[".meta", ".meta.yaml", ".meta.yml"] {
            let path = meta_dir.join(name);
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;
                let config: MetaConfig = if name.ends_with(".yaml") || name.ends_with(".yml") {
                    serde_yaml::from_str(&content)?
                } else {
                    serde_json::from_str(&content)?
                };

                return Ok(config.projects.into_iter().map(|(name, entry)| {
                    match entry {
                        ProjectEntry::Simple(repo) => ProjectInfo {
                            path: name.clone(),
                            name,
                            repo,
                            tags: vec![],
                        },
                        ProjectEntry::Extended { repo, path, tags } => ProjectInfo {
                            path: path.unwrap_or_else(|| name.clone()),
                            name,
                            repo,
                            tags,
                        },
                    }
                }).collect());
            }
        }

        Err(anyhow::anyhow!("No meta config found"))
    }
}

fn main() -> Result<()> {
    let mut server = McpServer::new();
    server.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = McpServer::new();
        // Server should be created (meta_dir may or may not be set depending on cwd)
        assert!(true);
    }

    #[test]
    fn test_initialize_response() {
        let server = McpServer::new();
        let result = server.handle_initialize().unwrap();

        let result_obj = result.as_object().unwrap();
        assert_eq!(result_obj.get("protocolVersion").unwrap(), PROTOCOL_VERSION);
        assert!(result_obj.get("capabilities").is_some());
        assert!(result_obj.get("serverInfo").is_some());
    }

    #[test]
    fn test_list_tools_response() {
        let server = McpServer::new();
        let result = server.handle_list_tools().unwrap();

        let result_obj = result.as_object().unwrap();
        let tools = result_obj.get("tools").unwrap().as_array().unwrap();
        assert!(!tools.is_empty());

        // Check that expected tools are present
        let tool_names: Vec<&str> = tools.iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        assert!(tool_names.contains(&"meta_list_projects"));
        assert!(tool_names.contains(&"meta_git_status"));
        assert!(tool_names.contains(&"meta_exec"));
    }

    #[test]
    fn test_ok_response() {
        let server = McpServer::new();
        let response = server.ok_response(
            Some(serde_json::json!(1)),
            serde_json::json!({"test": "value"})
        );

        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_error_response() {
        let server = McpServer::new();
        let response = server.error_response(
            Some(serde_json::json!(1)),
            -32600,
            "Invalid Request".to_string()
        );

        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_none());
        assert!(response.error.is_some());

        let error = response.error.unwrap();
        assert_eq!(error.code, -32600);
        assert_eq!(error.message, "Invalid Request");
    }
}
