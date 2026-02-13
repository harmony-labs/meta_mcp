//! MCP Server for meta - exposes multi-repo operations to AI tools.
//!
//! This server implements the Model Context Protocol (MCP) to allow AI assistants
//! like Claude to interact with meta repositories.

use anyhow::{Context, Result};
use meta_core::config::{self, ProjectInfo};
use meta_cli::dependency_graph::{self, ProjectDependencies};
use meta_cli::git_utils;
use meta_cli::query::{Query, RepoState, WorkspaceState};
use serde::{Deserialize, Serialize};
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
            .and_then(|dir| config::find_meta_config(&dir, None))
            .map(|(config_path, _)| config_path.parent().unwrap().to_path_buf());

        Self { meta_dir }
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
                    eprintln!("Failed to parse request: {e}");
                    continue;
                }
            };

            let response = self.handle_request(&request);
            let response_json = serde_json::to_string(&response)?;
            writeln!(stdout, "{response_json}")?;
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

    fn ok_response(
        &self,
        id: Option<serde_json::Value>,
        result: serde_json::Value,
    ) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error_response(
        &self,
        id: Option<serde_json::Value>,
        code: i32,
        message: String,
    ) -> JsonRpcResponse {
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
                tools: ToolsCapability {
                    list_changed: false,
                },
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
            // ================================================================
            // Core Tools
            // ================================================================
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
            Tool {
                name: "meta_get_config".to_string(),
                description: "Get the meta repository configuration including all projects, tags, and settings".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            Tool {
                name: "meta_get_project_path".to_string(),
                description: "Get the absolute path for a specific project".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Name of the project"
                        }
                    },
                    "required": ["project"]
                }),
            },
            // ================================================================
            // Multi-Repo Git Tools (Phase 5.1)
            // ================================================================
            Tool {
                name: "meta_git_status".to_string(),
                description: "Get git status for all projects with structured output showing dirty/clean state, branch, and changes".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Specific project to check (optional, defaults to all)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_git_pull".to_string(),
                description: "Pull changes from remote for all projects or filtered by tag".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        },
                        "rebase": {
                            "type": "boolean",
                            "description": "Use rebase instead of merge (default: false)"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_git_push".to_string(),
                description: "Push commits to remote for all projects or filtered by tag".to_string(),
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
                name: "meta_git_fetch".to_string(),
                description: "Fetch from remotes for all projects in parallel".to_string(),
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
                name: "meta_git_diff".to_string(),
                description: "Get diffs across repositories showing what has changed".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Specific project to diff (optional, defaults to all)"
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "Show only staged changes (default: false shows unstaged)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_git_branch".to_string(),
                description: "Get branch information for all projects including current branch, tracking branch, and ahead/behind status".to_string(),
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
                name: "meta_git_add".to_string(),
                description: "Stage files across repositories".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Specific project to add files in (optional, defaults to all)"
                        },
                        "files": {
                            "type": "string",
                            "description": "Files to add (default: '.' for all changed files)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_git_commit".to_string(),
                description: "Commit staged changes across repositories with a shared message".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Commit message"
                        },
                        "project": {
                            "type": "string",
                            "description": "Specific project to commit in (optional, defaults to all with staged changes)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    },
                    "required": ["message"]
                }),
            },
            Tool {
                name: "meta_git_checkout".to_string(),
                description: "Checkout a branch across repositories".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "branch": {
                            "type": "string",
                            "description": "Branch name to checkout"
                        },
                        "create": {
                            "type": "boolean",
                            "description": "Create the branch if it doesn't exist (default: false)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    },
                    "required": ["branch"]
                }),
            },
            Tool {
                name: "meta_git_multi_commit".to_string(),
                description: "Create commits with different messages for each repository. Allows tailored commit messages per project.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "commits": {
                            "type": "array",
                            "description": "Array of commit objects, each specifying a project and message",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "project": {
                                        "type": "string",
                                        "description": "Project name (use '.' for root repo)"
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Commit message for this project"
                                    }
                                },
                                "required": ["project", "message"]
                            }
                        }
                    },
                    "required": ["commits"]
                }),
            },
            // ================================================================
            // Build/Test Orchestration Tools (Phase 5.2)
            // ================================================================
            Tool {
                name: "meta_detect_build_systems".to_string(),
                description: "Detect build systems (Cargo, npm, make, etc.) for each project".to_string(),
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
                name: "meta_run_tests".to_string(),
                description: "Run tests across all projects using detected build systems".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        },
                        "project": {
                            "type": "string",
                            "description": "Specific project to test"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_build".to_string(),
                description: "Build all projects using detected build systems".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        },
                        "release": {
                            "type": "boolean",
                            "description": "Build in release mode (default: false)"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_clean".to_string(),
                description: "Clean build artifacts across all projects".to_string(),
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
            // ================================================================
            // Project Discovery & Analysis Tools (Phase 5.3)
            // ================================================================
            Tool {
                name: "meta_search_code".to_string(),
                description: "Search for patterns across all repositories using grep".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex supported)"
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "File glob pattern to filter (e.g., '*.rs', '*.ts')"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    },
                    "required": ["pattern"]
                }),
            },
            Tool {
                name: "meta_get_file_tree".to_string(),
                description: "Get file tree structure for projects".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Specific project to get tree for (optional, defaults to all)"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3)"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_list_plugins".to_string(),
                description: "List all installed meta plugins and their commands".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            // ================================================================
            // AI-Dominance Tools (Phase 9)
            // ================================================================
            Tool {
                name: "meta_query_repos".to_string(),
                description: "Query repositories by state/criteria using a simple DSL. Examples: 'dirty:true', 'tag:backend', 'dirty:true AND branch:main', 'modified_in:24h'".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query string using DSL (e.g., 'dirty:true AND tag:backend')"
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: "meta_workspace_state".to_string(),
                description: "Get a summary of the entire workspace state including dirty/clean counts, branches, and tags".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            Tool {
                name: "meta_analyze_impact".to_string(),
                description: "Analyze what would be affected if a project changes. Returns direct and transitive dependents.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project name to analyze impact for"
                        }
                    },
                    "required": ["project"]
                }),
            },
            Tool {
                name: "meta_execution_order".to_string(),
                description: "Get topological execution order respecting dependencies. Dependencies come before dependents.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag (optional)"
                        }
                    }
                }),
            },
            Tool {
                name: "meta_snapshot_create".to_string(),
                description: "Create a snapshot of the current workspace state for later rollback".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the snapshot"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description"
                        }
                    },
                    "required": ["name"]
                }),
            },
            Tool {
                name: "meta_snapshot_list".to_string(),
                description: "List all available workspace snapshots".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            Tool {
                name: "meta_snapshot_restore".to_string(),
                description: "Restore workspace to a previously saved snapshot".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the snapshot to restore"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force restore even if there are uncommitted changes (default: false)"
                        }
                    },
                    "required": ["name"]
                }),
            },
            Tool {
                name: "meta_batch_execute".to_string(),
                description: "Execute a command across projects with optional atomic rollback on failure".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute in each project"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter projects by tag (optional)"
                        },
                        "atomic": {
                            "type": "boolean",
                            "description": "If true, automatically rollback all projects if any fail (default: false)"
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
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        let result = match name {
            // Core tools
            "meta_list_projects" => self.tool_list_projects(&arguments),
            "meta_exec" => self.tool_exec(&arguments),
            "meta_get_config" => self.tool_get_config(&arguments),
            "meta_get_project_path" => self.tool_get_project_path(&arguments),
            // Git tools
            "meta_git_status" => self.tool_git_status(&arguments),
            "meta_git_pull" => self.tool_git_pull(&arguments),
            "meta_git_push" => self.tool_git_push(&arguments),
            "meta_git_fetch" => self.tool_git_fetch(&arguments),
            "meta_git_diff" => self.tool_git_diff(&arguments),
            "meta_git_branch" => self.tool_git_branch(&arguments),
            "meta_git_add" => self.tool_git_add(&arguments),
            "meta_git_commit" => self.tool_git_commit(&arguments),
            "meta_git_checkout" => self.tool_git_checkout(&arguments),
            "meta_git_multi_commit" => self.tool_git_multi_commit(&arguments),
            // Build/test tools
            "meta_detect_build_systems" => self.tool_detect_build_systems(&arguments),
            "meta_run_tests" => self.tool_run_tests(&arguments),
            "meta_build" => self.tool_build(&arguments),
            "meta_clean" => self.tool_clean(&arguments),
            // Discovery tools
            "meta_search_code" => self.tool_search_code(&arguments),
            "meta_get_file_tree" => self.tool_get_file_tree(&arguments),
            "meta_list_plugins" => self.tool_list_plugins(&arguments),
            // AI-Dominance tools
            "meta_query_repos" => self.tool_query_repos(&arguments),
            "meta_workspace_state" => self.tool_workspace_state(&arguments),
            "meta_analyze_impact" => self.tool_analyze_impact(&arguments),
            "meta_execution_order" => self.tool_execution_order(&arguments),
            "meta_snapshot_create" => self.tool_snapshot_create(&arguments),
            "meta_snapshot_list" => self.tool_snapshot_list(&arguments),
            "meta_snapshot_restore" => self.tool_snapshot_restore(&arguments),
            "meta_batch_execute" => self.tool_batch_execute(&arguments),
            _ => Err(anyhow::anyhow!("Unknown tool: {name}")),
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
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;

        // Filter by tag if specified
        let tag_filter = args.get("tag").and_then(|v| v.as_str());
        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        let mut output = format!(
            "Found {} project(s) in {}\n\n",
            filtered.len(),
            meta_dir.display()
        );
        for project in filtered {
            output.push_str(&format!("- {} ({})\n", project.name, project.path));
            if !project.tags.is_empty() {
                output.push_str(&format!("  Tags: {}\n", project.tags.join(", ")));
            }
        }

        Ok(output)
    }

    fn tool_git_status(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let project_filter = args.get("project").and_then(|v| v.as_str());

        // Run meta git status with --json for structured output
        let mut cmd = Command::new("meta");
        cmd.arg("--json").arg("git").arg("status");
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git status")?;

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
                        return Err(anyhow::anyhow!("Project '{project}' not found"));
                    }
                }
            }

            Ok(stdout.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("meta git status failed: {stderr}"))
        }
    }

    fn tool_exec(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let command = args
            .get("command")
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

        let output = cmd.output().context("Failed to execute meta command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("Command failed:\n{stdout}\n{stderr}"))
        }
    }

    // ========================================================================
    // Core Tools
    // ========================================================================

    fn tool_get_config(&self, _args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        // Find and return the raw config
        for name in &[".meta", ".meta.yaml", ".meta.yml"] {
            let path = meta_dir.join(name);
            if path.exists() {
                let content = std::fs::read_to_string(&path)?;
                return Ok(format!("Config file: {}\n\n{}", path.display(), content));
            }
        }

        Err(anyhow::anyhow!("No meta config found"))
    }

    fn tool_get_project_path(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let project_name = args
            .get("project")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'project' argument"))?;

        let projects = self.load_projects(meta_dir)?;

        for project in projects {
            if project.name == project_name {
                let full_path = meta_dir.join(&project.path);
                return Ok(serde_json::json!({
                    "project": project.name,
                    "path": full_path.display().to_string(),
                    "exists": full_path.exists()
                })
                .to_string());
            }
        }

        Err(anyhow::anyhow!("Project '{project_name}' not found"))
    }

    // ========================================================================
    // Git Tools
    // ========================================================================

    fn tool_git_pull(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("pull");

        if args
            .get("rebase")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            cmd.arg("--rebase");
        }

        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git pull")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("git pull failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_git_push(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("push");
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git push")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("git push failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_git_fetch(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("fetch");
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git fetch")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("git fetch failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_git_diff(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let project_filter = args.get("project").and_then(|v| v.as_str());
        let tag_filter = args.get("tag").and_then(|v| v.as_str());
        let staged = args
            .get("staged")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let filtered: Vec<&ProjectInfo> = projects
            .iter()
            .filter(|p| {
                if let Some(project) = project_filter {
                    return p.name == project;
                }
                if let Some(tag) = tag_filter {
                    return p.tags.contains(&tag.to_string());
                }
                true
            })
            .collect();

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let mut cmd = Command::new("git");
            cmd.arg("diff");
            if staged {
                cmd.arg("--staged");
            }
            cmd.current_dir(&project_path);

            let output = cmd.output()?;
            let diff = String::from_utf8_lossy(&output.stdout);

            if !diff.is_empty() {
                results.push(serde_json::json!({
                    "project": project.name,
                    "diff": diff.to_string()
                }));
            }
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn tool_git_branch(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            // Get current branch
            let current_branch =
                git_utils::current_branch(&project_path).unwrap_or_else(|| "HEAD".to_string());

            // Get tracking branch info
            let tracking_output = Command::new("git")
                .args([
                    "for-each-ref",
                    "--format=%(upstream:short)",
                    &format!("refs/heads/{current_branch}"),
                ])
                .current_dir(&project_path)
                .output()?;
            let tracking_branch = String::from_utf8_lossy(&tracking_output.stdout)
                .trim()
                .to_string();

            // Get ahead/behind counts
            let mut ahead = 0;
            let mut behind = 0;
            if !tracking_branch.is_empty() {
                let ahead_behind = Command::new("git")
                    .args([
                        "rev-list",
                        "--left-right",
                        "--count",
                        &format!("{current_branch}...{tracking_branch}"),
                    ])
                    .current_dir(&project_path)
                    .output()?;
                let counts = String::from_utf8_lossy(&ahead_behind.stdout);
                let parts: Vec<&str> = counts.trim().split('\t').collect();
                if parts.len() == 2 {
                    ahead = parts[0].parse().unwrap_or(0);
                    behind = parts[1].parse().unwrap_or(0);
                }
            }

            results.push(serde_json::json!({
                "project": project.name,
                "branch": current_branch,
                "tracking": if tracking_branch.is_empty() { None } else { Some(&tracking_branch) },
                "ahead": ahead,
                "behind": behind
            }));
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn tool_git_add(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let files = args.get("files").and_then(|v| v.as_str()).unwrap_or(".");

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("add").arg(files);
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git add")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(format!("Staged files: {files}\n{stdout}"))
        } else {
            Err(anyhow::anyhow!("git add failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_git_commit(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'message' argument"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("commit").arg("-m").arg(message);
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta git commit")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("git commit failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_git_multi_commit(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let commits = args
            .get("commits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'commits' argument"))?;

        #[derive(Debug, Serialize)]
        struct CommitResult {
            project: String,
            success: bool,
            message: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            error: Option<String>,
        }

        let mut results: Vec<CommitResult> = Vec::new();

        for commit_obj in commits {
            let project = commit_obj
                .get("project")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing 'project' in commit entry"))?;

            let message = commit_obj
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing 'message' in commit entry"))?;

            // Determine the path for this project
            let project_path = if project == "." {
                meta_dir.clone()
            } else {
                meta_dir.join(project)
            };

            if !project_path.exists() {
                results.push(CommitResult {
                    project: project.to_string(),
                    success: false,
                    message: message.to_string(),
                    error: Some(format!(
                        "Project path does not exist: {}",
                        project_path.display()
                    )),
                });
                continue;
            }

            // Execute git commit for this project
            let output = Command::new("git")
                .arg("-C")
                .arg(&project_path)
                .arg("commit")
                .arg("-m")
                .arg(message)
                .output();

            match output {
                Ok(out) => {
                    if out.status.success() {
                        results.push(CommitResult {
                            project: project.to_string(),
                            success: true,
                            message: message.to_string(),
                            error: None,
                        });
                    } else {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        results.push(CommitResult {
                            project: project.to_string(),
                            success: false,
                            message: message.to_string(),
                            error: Some(stderr.trim().to_string()),
                        });
                    }
                }
                Err(e) => {
                    results.push(CommitResult {
                        project: project.to_string(),
                        success: false,
                        message: message.to_string(),
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let succeeded = results.iter().filter(|r| r.success).count();
        let failed = results.iter().filter(|r| !r.success).count();

        let output = serde_json::json!({
            "results": results,
            "summary": {
                "total": results.len(),
                "succeeded": succeeded,
                "failed": failed
            }
        });

        Ok(serde_json::to_string_pretty(&output)?)
    }

    fn tool_git_checkout(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let branch = args
            .get("branch")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'branch' argument"))?;

        let create = args
            .get("create")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        cmd.arg("git").arg("checkout");
        if create {
            cmd.arg("-b");
        }
        cmd.arg(branch);
        cmd.current_dir(meta_dir);

        let output = cmd
            .output()
            .context("Failed to execute meta git checkout")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("git checkout failed:\n{stdout}\n{stderr}"))
        }
    }

    // ========================================================================
    // Build/Test Tools
    // ========================================================================

    fn tool_detect_build_systems(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let mut build_systems = Vec::new();

            // Check for various build systems
            if project_path.join("Cargo.toml").exists() {
                build_systems.push("cargo");
            }
            if project_path.join("package.json").exists() {
                build_systems.push("npm");
            }
            if project_path.join("Makefile").exists() || project_path.join("makefile").exists() {
                build_systems.push("make");
            }
            if project_path.join("go.mod").exists() {
                build_systems.push("go");
            }
            if project_path.join("pom.xml").exists() {
                build_systems.push("maven");
            }
            if project_path.join("build.gradle").exists()
                || project_path.join("build.gradle.kts").exists()
            {
                build_systems.push("gradle");
            }
            if project_path.join("pyproject.toml").exists()
                || project_path.join("setup.py").exists()
            {
                build_systems.push("python");
            }

            results.push(serde_json::json!({
                "project": project.name,
                "path": project.path,
                "build_systems": build_systems
            }));
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn tool_run_tests(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let project_filter = args.get("project").and_then(|v| v.as_str());
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = projects
            .iter()
            .filter(|p| {
                if let Some(project) = project_filter {
                    return p.name == project;
                }
                if let Some(tag) = tag_filter {
                    return p.tags.contains(&tag.to_string());
                }
                true
            })
            .collect();

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let (cmd_name, cmd_args): (&str, Vec<&str>) =
                if project_path.join("Cargo.toml").exists() {
                    ("cargo", vec!["test"])
                } else if project_path.join("package.json").exists() {
                    ("npm", vec!["test"])
                } else if project_path.join("go.mod").exists() {
                    ("go", vec!["test", "./..."])
                } else if project_path.join("Makefile").exists() {
                    ("make", vec!["test"])
                } else {
                    continue; // No recognized test command
                };

            let output = Command::new(cmd_name)
                .args(&cmd_args)
                .current_dir(&project_path)
                .output();

            match output {
                Ok(out) => {
                    results.push(serde_json::json!({
                        "project": project.name,
                        "command": format!("{} {}", cmd_name, cmd_args.join(" ")),
                        "success": out.status.success(),
                        "stdout": String::from_utf8_lossy(&out.stdout).to_string(),
                        "stderr": String::from_utf8_lossy(&out.stderr).to_string()
                    }));
                }
                Err(e) => {
                    results.push(serde_json::json!({
                        "project": project.name,
                        "error": e.to_string()
                    }));
                }
            }
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn tool_build(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let release = args
            .get("release")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut cmd = Command::new("meta");
        cmd.arg("--json");

        if let Some(tag) = args.get("tag").and_then(|v| v.as_str()) {
            cmd.arg("--tag").arg(tag);
        }

        // Use meta exec to run build commands
        if release {
            cmd.args(["exec", "--", "cargo", "build", "--release"]);
        } else {
            cmd.args(["exec", "--", "cargo", "build"]);
        }

        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to execute meta build")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Err(anyhow::anyhow!("build failed:\n{stdout}\n{stderr}"))
        }
    }

    fn tool_clean(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let (cmd_name, cmd_args): (&str, Vec<&str>) =
                if project_path.join("Cargo.toml").exists() {
                    ("cargo", vec!["clean"])
                } else if project_path.join("package.json").exists() {
                    // For npm projects, we'd typically remove node_modules
                    continue; // Skip npm for now as rm -rf is dangerous
                } else if project_path.join("go.mod").exists() {
                    ("go", vec!["clean"])
                } else if project_path.join("Makefile").exists() {
                    ("make", vec!["clean"])
                } else {
                    continue;
                };

            let output = Command::new(cmd_name)
                .args(&cmd_args)
                .current_dir(&project_path)
                .output();

            match output {
                Ok(out) => {
                    results.push(serde_json::json!({
                        "project": project.name,
                        "command": format!("{} {}", cmd_name, cmd_args.join(" ")),
                        "success": out.status.success()
                    }));
                }
                Err(e) => {
                    results.push(serde_json::json!({
                        "project": project.name,
                        "error": e.to_string()
                    }));
                }
            }
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    // ========================================================================
    // Discovery Tools
    // ========================================================================

    fn tool_search_code(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'pattern' argument"))?;

        let file_pattern = args.get("file_pattern").and_then(|v| v.as_str());

        let projects = self.load_projects(meta_dir)?;
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let mut cmd = Command::new("grep");
            cmd.args(["-r", "-n", "-I"]); // recursive, line numbers, skip binary

            if let Some(fp) = file_pattern {
                cmd.args(["--include", fp]);
            }

            cmd.arg(pattern);
            cmd.current_dir(&project_path);

            let output = cmd.output();

            match output {
                Ok(out) => {
                    let matches = String::from_utf8_lossy(&out.stdout);
                    if !matches.is_empty() {
                        results.push(serde_json::json!({
                            "project": project.name,
                            "matches": matches.lines().take(50).collect::<Vec<_>>()
                        }));
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn tool_get_file_tree(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let depth = args.get("depth").and_then(|v| v.as_i64()).unwrap_or(3) as usize;

        let projects = self.load_projects(meta_dir)?;
        let project_filter = args.get("project").and_then(|v| v.as_str());
        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let filtered: Vec<&ProjectInfo> = projects
            .iter()
            .filter(|p| {
                if let Some(project) = project_filter {
                    return p.name == project;
                }
                if let Some(tag) = tag_filter {
                    return p.tags.contains(&tag.to_string());
                }
                true
            })
            .collect();

        let mut results = Vec::new();

        for project in filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let tree = Self::build_file_tree(&project_path, depth, 0)?;
            results.push(serde_json::json!({
                "project": project.name,
                "tree": tree
            }));
        }

        Ok(serde_json::to_string_pretty(&results)?)
    }

    fn build_file_tree(
        path: &std::path::Path,
        max_depth: usize,
        current_depth: usize,
    ) -> Result<serde_json::Value> {
        if current_depth >= max_depth {
            return Ok(serde_json::json!(null));
        }

        if path.is_file() {
            return Ok(serde_json::json!({
                "type": "file",
                "name": path.file_name().unwrap_or_default().to_string_lossy()
            }));
        }

        let mut children = Vec::new();

        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                let name = entry_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                // Skip hidden files and common ignored directories
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    continue;
                }

                if entry_path.is_dir() {
                    children.push(serde_json::json!({
                        "type": "directory",
                        "name": name,
                        "children": Self::build_file_tree(&entry_path, max_depth, current_depth + 1)?
                    }));
                } else {
                    children.push(serde_json::json!({
                        "type": "file",
                        "name": name
                    }));
                }
            }
        }

        Ok(serde_json::Value::Array(children))
    }

    fn tool_list_plugins(&self, _args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let mut cmd = Command::new("meta");
        cmd.arg("--json").arg("plugins").arg("list");
        cmd.current_dir(meta_dir);

        let output = cmd.output().context("Failed to list plugins")?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!(
                "Failed to list plugins:\n{stdout}\n{stderr}"
            ))
        }
    }

    // ========================================================================
    // AI-Dominance Tools (Phase 9)
    // ========================================================================

    fn tool_query_repos(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let query_str = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'query' argument"))?;

        let query = Query::parse(query_str)?;
        let projects = self.load_projects(meta_dir)?;
        let mut matching = Vec::new();

        for project in &projects {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }

            let state = RepoState::collect(&project.name, &project_path, &project.tags)?;
            if state.matches(&query) {
                matching.push(state);
            }
        }

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "query": query_str,
            "matches": matching.len(),
            "projects": matching
        }))?)
    }

    fn tool_workspace_state(&self, _args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let projects = self.load_projects(meta_dir)?;
        let mut repo_states = Vec::new();

        for project in &projects {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                continue;
            }
            if let Ok(state) = RepoState::collect(&project.name, &project_path, &project.tags) {
                repo_states.push(state);
            }
        }

        let workspace_state = WorkspaceState::from_repos(&repo_states);
        Ok(serde_json::to_string_pretty(&workspace_state)?)
    }

    fn tool_analyze_impact(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let project_name = args
            .get("project")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'project' argument"))?;

        let projects = self.load_project_dependencies(meta_dir)?;
        let graph = dependency_graph::DependencyGraph::build(projects)?;
        let impact = graph.analyze_impact(project_name);

        Ok(serde_json::to_string_pretty(&impact)?)
    }

    fn tool_execution_order(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let tag_filter = args.get("tag").and_then(|v| v.as_str());

        let projects = self.load_project_dependencies(meta_dir)?;
        let graph = dependency_graph::DependencyGraph::build(projects)?;
        let tags: Vec<String> = tag_filter.map(|t| vec![t.to_string()]).unwrap_or_default();
        let order = graph.execution_order_filtered(&tags)?;

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "execution_order": order,
            "count": order.len(),
            "tag_filter": tag_filter
        }))?)
    }

    fn tool_snapshot_create(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'name' argument"))?;

        let description = args.get("description").and_then(|v| v.as_str());

        let projects = self.load_projects(meta_dir)?;
        let mut project_snapshots = Vec::new();

        for project in &projects {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() || !project_path.join(".git").exists() {
                continue;
            }

            let branch =
                git_utils::current_branch(&project_path).unwrap_or_else(|| "unknown".to_string());
            let commit = self
                .git_output(&project_path, &["rev-parse", "HEAD"])
                .unwrap_or_else(|_| "unknown".to_string());
            let is_dirty = git_utils::is_dirty(&project_path).unwrap_or(false);

            project_snapshots.push(serde_json::json!({
                "name": project.name,
                "path": project.path,
                "branch": branch,
                "commit": commit,
                "is_dirty": is_dirty
            }));
        }

        let snapshot = serde_json::json!({
            "name": name,
            "created_at": chrono::Utc::now().to_rfc3339(),
            "description": description,
            "meta_dir": meta_dir.to_string_lossy(),
            "projects": project_snapshots
        });

        // Save snapshot
        let snapshots_dir = meta_dir.join(".meta-snapshots");
        std::fs::create_dir_all(&snapshots_dir)?;
        let filename = format!("{}.json", name.replace(['/', '\\', ' '], "_"));
        let snapshot_path = snapshots_dir.join(&filename);
        std::fs::write(&snapshot_path, serde_json::to_string_pretty(&snapshot)?)?;

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "status": "created",
            "name": name,
            "path": snapshot_path.to_string_lossy(),
            "projects_count": project_snapshots.len()
        }))?)
    }

    fn tool_snapshot_list(&self, _args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let snapshots_dir = meta_dir.join(".meta-snapshots");
        let mut snapshots = Vec::new();

        if snapshots_dir.exists() {
            for entry in std::fs::read_dir(&snapshots_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(snapshot) = serde_json::from_str::<serde_json::Value>(&content) {
                            snapshots.push(serde_json::json!({
                                "name": snapshot.get("name"),
                                "created_at": snapshot.get("created_at"),
                                "description": snapshot.get("description"),
                                "projects_count": snapshot.get("projects").and_then(|p| p.as_array()).map(|a| a.len())
                            }));
                        }
                    }
                }
            }
        }

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "snapshots": snapshots,
            "count": snapshots.len()
        }))?)
    }

    fn tool_snapshot_restore(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'name' argument"))?;

        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        // Load snapshot
        let snapshots_dir = meta_dir.join(".meta-snapshots");
        let filename = format!("{}.json", name.replace(['/', '\\', ' '], "_"));
        let snapshot_path = snapshots_dir.join(&filename);

        if !snapshot_path.exists() {
            return Err(anyhow::anyhow!("Snapshot '{name}' not found"));
        }

        let content = std::fs::read_to_string(&snapshot_path)?;
        let snapshot: serde_json::Value = serde_json::from_str(&content)?;

        let projects = snapshot
            .get("projects")
            .and_then(|p| p.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid snapshot format"))?;

        let mut restored = Vec::new();
        let mut failed = Vec::new();

        for project in projects {
            let proj_name = project
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let proj_path = project.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let branch = project
                .get("branch")
                .and_then(|v| v.as_str())
                .unwrap_or("main");
            let commit = project.get("commit").and_then(|v| v.as_str()).unwrap_or("");

            let full_path = meta_dir.join(proj_path);
            if !full_path.exists() {
                failed.push(serde_json::json!({
                    "project": proj_name,
                    "error": "Path does not exist"
                }));
                continue;
            }

            // Check if dirty and not force
            let is_dirty = git_utils::is_dirty(&full_path).unwrap_or(false);
            if is_dirty && !force {
                failed.push(serde_json::json!({
                    "project": proj_name,
                    "error": "Has uncommitted changes (use force=true to override)"
                }));
                continue;
            }

            // Stash if dirty and force
            if is_dirty && force {
                let _ =
                    self.git_command(&full_path, &["stash", "push", "-m", "meta-restore-backup"]);
            }

            // Checkout branch and reset
            if let Err(e) = self.git_command(&full_path, &["checkout", branch]) {
                failed.push(serde_json::json!({
                    "project": proj_name,
                    "error": format!("Failed to checkout: {}", e)
                }));
                continue;
            }

            if let Err(e) = self.git_command(&full_path, &["reset", "--hard", commit]) {
                failed.push(serde_json::json!({
                    "project": proj_name,
                    "error": format!("Failed to reset: {}", e)
                }));
                continue;
            }

            restored.push(proj_name.to_string());
        }

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "status": if failed.is_empty() { "success" } else { "partial" },
            "restored": restored,
            "failed": failed,
            "restored_count": restored.len(),
            "failed_count": failed.len()
        }))?)
    }

    fn tool_batch_execute(&self, args: &serde_json::Value) -> Result<String> {
        let meta_dir = self
            .meta_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No meta repository found"))?;

        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;

        let tag_filter = args.get("tag").and_then(|v| v.as_str());
        let atomic = args
            .get("atomic")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let projects = self.load_projects(meta_dir)?;

        // Filter by tag
        let filtered: Vec<&ProjectInfo> = if let Some(tag) = tag_filter {
            projects
                .iter()
                .filter(|p| p.tags.contains(&tag.to_string()))
                .collect()
        } else {
            projects.iter().collect()
        };

        // Create pre-execution snapshot if atomic
        let snapshot_name = if atomic {
            let name = format!("atomic-batch-{}", chrono::Utc::now().timestamp());
            let snapshot_args = serde_json::json!({
                "name": name,
                "description": "Automatic snapshot before atomic batch execution"
            });
            let _ = self.tool_snapshot_create(&snapshot_args);
            Some(name)
        } else {
            None
        };

        let mut results = Vec::new();
        let mut has_failure = false;

        for project in &filtered {
            let project_path = meta_dir.join(&project.path);
            if !project_path.exists() {
                results.push(serde_json::json!({
                    "project": project.name,
                    "success": false,
                    "error": "Path does not exist"
                }));
                has_failure = true;
                continue;
            }

            let output = Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&project_path)
                .output();

            match output {
                Ok(out) => {
                    let success = out.status.success();
                    if !success {
                        has_failure = true;
                    }
                    results.push(serde_json::json!({
                        "project": project.name,
                        "success": success,
                        "stdout": String::from_utf8_lossy(&out.stdout).to_string(),
                        "stderr": String::from_utf8_lossy(&out.stderr).to_string()
                    }));
                }
                Err(e) => {
                    has_failure = true;
                    results.push(serde_json::json!({
                        "project": project.name,
                        "success": false,
                        "error": e.to_string()
                    }));
                }
            }

            // If atomic and failure, stop and rollback
            if atomic && has_failure {
                break;
            }
        }

        // Rollback if atomic and failure
        let mut rollback_result = None;
        if atomic && has_failure {
            if let Some(ref snapshot_name) = snapshot_name {
                let restore_args = serde_json::json!({
                    "name": snapshot_name,
                    "force": true
                });
                rollback_result = Some(self.tool_snapshot_restore(&restore_args)?);
            }
        }

        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "command": command,
            "results": results,
            "has_failure": has_failure,
            "rolled_back": rollback_result.is_some(),
            "rollback_result": rollback_result
        }))?)
    }

    fn git_output(&self, path: &std::path::Path, args: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .with_context(|| format!("Failed to run git {args:?}"))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(anyhow::anyhow!("Git command failed"))
        }
    }

    fn git_command(&self, path: &std::path::Path, args: &[&str]) -> Result<()> {
        let output = Command::new("git")
            .args(args)
            .current_dir(path)
            .output()
            .with_context(|| format!("Failed to run git {args:?}"))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Git command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }

    // ========================================================================
    // Dependency Graph Helpers
    // ========================================================================

    fn load_project_dependencies(
        &self,
        meta_dir: &std::path::Path,
    ) -> Result<Vec<ProjectDependencies>> {
        let projects = self.load_projects(meta_dir)?;
        Ok(projects.into_iter().map(|p| p.into()).collect())
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn load_projects(&self, meta_dir: &std::path::Path) -> Result<Vec<ProjectInfo>> {
        let (config_path, _) = config::find_meta_config(meta_dir, None)
            .ok_or_else(|| anyhow::anyhow!("No meta config found"))?;
        let (projects, _ignore) = config::parse_meta_config(&config_path)?;
        Ok(projects)
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
        // We just verify it doesn't panic and returns a valid server
        drop(server);
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
        let tool_names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();

        // Core tools
        assert!(tool_names.contains(&"meta_list_projects"));
        assert!(tool_names.contains(&"meta_exec"));
        assert!(tool_names.contains(&"meta_get_config"));
        assert!(tool_names.contains(&"meta_get_project_path"));

        // Git tools
        assert!(tool_names.contains(&"meta_git_status"));
        assert!(tool_names.contains(&"meta_git_pull"));
        assert!(tool_names.contains(&"meta_git_push"));
        assert!(tool_names.contains(&"meta_git_fetch"));
        assert!(tool_names.contains(&"meta_git_diff"));
        assert!(tool_names.contains(&"meta_git_branch"));
        assert!(tool_names.contains(&"meta_git_add"));
        assert!(tool_names.contains(&"meta_git_commit"));
        assert!(tool_names.contains(&"meta_git_checkout"));
        assert!(tool_names.contains(&"meta_git_multi_commit"));

        // Build/test tools
        assert!(tool_names.contains(&"meta_detect_build_systems"));
        assert!(tool_names.contains(&"meta_run_tests"));
        assert!(tool_names.contains(&"meta_build"));
        assert!(tool_names.contains(&"meta_clean"));

        // Discovery tools
        assert!(tool_names.contains(&"meta_search_code"));
        assert!(tool_names.contains(&"meta_get_file_tree"));
        assert!(tool_names.contains(&"meta_list_plugins"));

        // AI-Dominance tools
        assert!(tool_names.contains(&"meta_query_repos"));
        assert!(tool_names.contains(&"meta_workspace_state"));
        assert!(tool_names.contains(&"meta_analyze_impact"));
        assert!(tool_names.contains(&"meta_execution_order"));
        assert!(tool_names.contains(&"meta_snapshot_create"));
        assert!(tool_names.contains(&"meta_snapshot_list"));
        assert!(tool_names.contains(&"meta_snapshot_restore"));
        assert!(tool_names.contains(&"meta_batch_execute"));

        // Verify total count (4 core + 10 git + 4 build + 3 discovery + 8 AI = 29)
        assert_eq!(tool_names.len(), 29);
    }

    #[test]
    fn test_ok_response() {
        let server = McpServer::new();
        let response = server.ok_response(
            Some(serde_json::json!(1)),
            serde_json::json!({"test": "value"}),
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
            "Invalid Request".to_string(),
        );

        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_none());
        assert!(response.error.is_some());

        let error = response.error.unwrap();
        assert_eq!(error.code, -32600);
        assert_eq!(error.message, "Invalid Request");
    }

    #[test]
    fn test_multi_commit_tool_schema() {
        let server = McpServer::new();
        let result = server.handle_list_tools().unwrap();

        let result_obj = result.as_object().unwrap();
        let tools = result_obj.get("tools").unwrap().as_array().unwrap();

        // Find meta_git_multi_commit tool
        let multi_commit_tool = tools
            .iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("meta_git_multi_commit"))
            .expect("meta_git_multi_commit tool should exist");

        // Verify schema structure
        let schema = multi_commit_tool.get("inputSchema").unwrap();
        let props = schema.get("properties").unwrap().as_object().unwrap();

        // Should have a commits property
        assert!(props.contains_key("commits"));

        let commits_prop = props.get("commits").unwrap();
        assert_eq!(commits_prop.get("type").unwrap(), "array");

        // Verify items schema
        let items = commits_prop.get("items").unwrap();
        let item_props = items.get("properties").unwrap().as_object().unwrap();
        assert!(item_props.contains_key("project"));
        assert!(item_props.contains_key("message"));

        // Verify required fields in items
        let item_required = items.get("required").unwrap().as_array().unwrap();
        let required_fields: Vec<&str> = item_required.iter().filter_map(|v| v.as_str()).collect();
        assert!(required_fields.contains(&"project"));
        assert!(required_fields.contains(&"message"));
    }

    #[test]
    fn test_multi_commit_missing_commits_arg() {
        let server = McpServer::new();

        // Test with empty args - should fail with missing commits
        let result = server.tool_git_multi_commit(&serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("commits"));
    }

    #[test]
    fn test_multi_commit_invalid_commits_format() {
        let server = McpServer::new();

        // Test with commits as string instead of array
        let result = server.tool_git_multi_commit(&serde_json::json!({
            "commits": "not an array"
        }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("commits"));
    }

    #[test]
    fn test_multi_commit_missing_project_in_entry() {
        let server = McpServer::new();

        // Test with commit entry missing project
        let result = server.tool_git_multi_commit(&serde_json::json!({
            "commits": [
                {"message": "Test commit"}
            ]
        }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("project"));
    }

    #[test]
    fn test_multi_commit_missing_message_in_entry() {
        let server = McpServer::new();

        // Test with commit entry missing message
        let result = server.tool_git_multi_commit(&serde_json::json!({
            "commits": [
                {"project": "test-repo"}
            ]
        }));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("message"));
    }
}
