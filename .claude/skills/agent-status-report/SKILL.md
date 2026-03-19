---
name: agent-status-report
description: >
  Generate a status report of all running and completed agent tasks (background
  subagents, background bash commands, etc.). Lists each agent's task ID, status,
  description, and output summary. Writes the report to a timestamped .txt file
  in the current working directory. Use periodically to monitor agent team progress,
  or when the user asks "what are the agents doing" / "show me agent status".
  Usage: /agent-status-report
tools: Bash,Read,Write,TaskList,TaskGet,TaskOutput
---

# Agent Status Report

Generate a report of all agent/task activity and write it to a .txt file.

## Step 1: Collect Task Information

### 1.1 List all tasks

Use the `TaskList` tool to get all tasks (running, completed, failed, etc.).

### 1.2 Get details for each task

For each task returned by TaskList, use `TaskGet` to retrieve:
- Task ID
- Status (in_progress, completed, failed, stopped)
- Description / goal
- Creation time

### 1.3 Get task output

For completed or in_progress tasks, use `TaskOutput` to retrieve the output
or partial output. Summarize long outputs to keep the report readable.

### 1.4 Check background commands

Also check for any background bash commands that may be running:

```bash
# List any background processes started by this session
jobs -l 2>/dev/null
```

---

## Step 2: Collect Git Activity (Optional)

If inside a git repository, gather recent commit activity as a proxy for
what agents have been working on:

```bash
# Recent commits in the last hour/day
git log --oneline --since="1 hour ago" --all 2>/dev/null
# Or broader
git log --oneline -10 --all 2>/dev/null
```

---

## Step 3: Generate Report

### Report format

```
================================================================
AGENT STATUS REPORT
Generated: YYYY-MM-DD HH:MM:SS
Working Directory: /path/to/cwd
================================================================

SUMMARY
-------
Total tasks:     N
  Running:       X
  Completed:     Y
  Failed:        Z
  Stopped:       W

TASK DETAILS
------------

[1] Task ID: <id>
    Status:      in_progress
    Description: <what this task is doing>
    Started:     <time>
    Output (latest):
      <last ~10 lines of output or summary>

[2] Task ID: <id>
    Status:      completed
    Description: <what this task did>
    Duration:    <time>
    Result:
      <summary of output, key findings, or return value>

[3] Task ID: <id>
    Status:      failed
    Description: <what this task attempted>
    Error:
      <error message or last output before failure>

...

RECENT ACTIVITY
---------------
<recent git commits or file changes if available>

NOTES
-----
<any observations: stalled tasks, tasks waiting for input, etc.>

================================================================
END OF REPORT
================================================================
```

### Summarization rules

- For task outputs longer than 20 lines, summarize to key points
- For benchmark results, extract the metric values
- For build/compile tasks, report success/failure and any errors
- For search/research tasks, list the key findings
- For running tasks, show the last few lines of output as progress indicator

---

## Step 4: Write Report File

Write the report to a timestamped .txt file in the current directory:

```python
filename = f"agent_report_{timestamp}.txt"
# Example: agent_report_20260309_134500.txt
```

Use the Write tool to create the file.

Also print the report content directly to the user so they can see it
immediately without opening the file.

---

## Step 5: Detect Issues

Flag any concerning patterns:

| Pattern | Alert |
|---------|-------|
| Task running > 10 minutes with no output | "Task [id] may be stalled" |
| Task failed with non-zero exit code | "Task [id] FAILED: <error>" |
| Multiple tasks doing similar work | "Possible duplicate work: [id1] and [id2]" |
| No tasks running or recent | "No active tasks found" |

---

## Handling No Tasks

If there are no tasks (TaskList returns empty):

1. Report "No agent tasks found"
2. Check for background bash processes as fallback
3. Check recent file modifications as activity indicator:
   ```bash
   find . -maxdepth 2 -name "*.py" -newer /tmp/.last_report -type f 2>/dev/null | head -20
   ```
4. If still nothing, write a minimal report stating no activity detected

---

## Periodic Usage

The user may want periodic reports. Since Claude Code cannot run timers,
suggest these patterns:

1. **Manual**: Run `/agent-status-report` whenever curious
2. **After task completion**: The user can ask "report status" after kicking off
   background work
3. **In CLAUDE.md**: Add a reminder like "run /agent-status-report after every
   major task completes" to automate the habit
