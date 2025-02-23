#!/usr/bin/env python3
"""
Interactive chat script with integrated tools for web search, content scraping, and package management.
Using a multi-agent architecture with Planner and Executor agents.
"""
import json
import shutil
import argparse
import logging
import os
import sys
import signal
from typing import Set, Optional, Dict, Any
from datetime import datetime

from planner_agent import PlannerAgent, PlannerContext
from executor_agent import ExecutorAgent, ExecutorContext
from common import TokenUsage, TokenTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(debug: bool) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Research Agent')
    parser.add_argument('query', help='The research query to process')
    parser.add_argument('--planner_model', default='gpt-4o', help='The OpenAI model to use for planner')
    parser.add_argument('--executor_model', default='gpt-4o', help='The OpenAI model to use for executor')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def parse_args_config() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deep Research Agent')
    parser.add_argument('--config', default='research_task.json', help='Path to the JSON configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        required_fields = ['query']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config file")

        # Set defaults if not provided
        config.setdefault('planner_model', 'gpt-4o-mini')
        config.setdefault('executor_model', 'gpt-4o-mini')

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {config_path}")

class AgentCommunication:
    """Handles structured communication between Planner and Executor agents."""

    @staticmethod
    def format_planner_instructions(
        task: str,
        deliverables: list,
        constraints: list,
        prerequisites: list
    ) -> str:
        """Format instructions from Planner to Executor."""
        return f"""PLANNER_INSTRUCTIONS:
- Task: {task}
- Required Deliverables:
  {chr(10).join(f'  - {d}' for d in deliverables)}
- Constraints:
  {chr(10).join(f'  - {c}' for c in constraints)}
- Prerequisites:
  {chr(10).join(f'  - {p}' for p in prerequisites)}"""

    @staticmethod
    def format_executor_feedback(
        status: str,
        completion_details: str,
        blockers: list,
        resources_needed: list,
        next_task_ready: bool
    ) -> str:
        """Format feedback from Executor to Planner."""
        return f"""EXECUTOR_FEEDBACK:
- Task Status: {status}
- Completion Details: {completion_details}
- Blockers:
  {chr(10).join(f'  - {b}' for b in blockers)}
- Resources Needed:
  {chr(10).join(f'  - {r}' for r in resources_needed)}
- Next Task Readiness: {'Ready' if next_task_ready else 'Not Ready'}"""

    @staticmethod
    def parse_planner_instructions(instructions: str) -> Dict[str, Any]:
        """Parse structured instructions from Planner."""
        # Basic parsing implementation
        sections = instructions.split('\n')
        result = {
            'task': '',
            'deliverables': [],
            'constraints': [],
            'prerequisites': []
        }
        current_section = None

        for line in sections:
            line = line.strip()
            if line.startswith('- Task:'):
                result['task'] = line[7:].strip()
            elif line.startswith('- Required Deliverables:'):
                current_section = 'deliverables'
            elif line.startswith('- Constraints:'):
                current_section = 'constraints'
            elif line.startswith('- Prerequisites:'):
                current_section = 'prerequisites'
            elif line.startswith('  - ') and current_section:
                result[current_section].append(line[4:])

        return result

    @staticmethod
    def parse_executor_feedback(feedback: str) -> Dict[str, Any]:
        """Parse structured feedback from Executor."""
        # Basic parsing implementation
        sections = feedback.split('\n')
        result = {
            'status': '',
            'completion_details': '',
            'blockers': [],
            'resources_needed': [],
            'next_task_ready': False
        }
        current_section = None

        for line in sections:
            line = line.strip()
            if line.startswith('- Task Status:'):
                result['status'] = line[13:].strip()
            elif line.startswith('- Completion Details:'):
                result['completion_details'] = line[20:].strip()
            elif line.startswith('- Blockers:'):
                current_section = 'blockers'
            elif line.startswith('- Resources Needed:'):
                current_section = 'resources_needed'
            elif line.startswith('- Next Task Readiness:'):
                result['next_task_ready'] = 'Ready' in line
            elif line.startswith('  - ') and current_section:
                result[current_section].append(line[4:])

        return result

class ResearchSession:
    """Manages the research session with Planner and Executor agents."""

    #def __init__(self, planner_model: str, executor_model: str, debug: bool = False):
    def __init__(self, planner_model: str, executor_model: str, report_name: str = None, debug: bool = False):
        """Initialize the research session.

        Args:
            planner_model: The OpenAI model to use for Planner agent
            executor_model: The OpenAI model to use for Executor agent
            report_name: Name for the research report and output folder
            debug: Whether to enable debug mode
        """
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.report_name = report_name if report_name else 'research_report'
        self.debug = debug
        self.round_count = 0

        # Set up debug logging if enabled
        if debug:
            logging.getLogger('tools').setLevel(logging.DEBUG)
            logging.getLogger('executor_agent').setLevel(logging.DEBUG)
            logging.getLogger('planner_agent').setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        self.planner = PlannerAgent(model=planner_model)
        self.executor = ExecutorAgent(model=executor_model)
        self.created_files: Set[str] = set()
        self.token_tracker = TokenTracker()
        self.agent_communication = AgentCommunication()

        # Initialize scratchpad with required sections
        self._initialize_scratchpad()
        self.created_files.add('scratchpad.md')
        logger.info("Created scratchpad.md with initial sections")

    def _initialize_scratchpad(self) -> None:
        """Initialize scratchpad.md with the required sections."""
        initial_content = """### Background and Motivation
(Planner writes: User/business requirements, macro objectives, why this problem needs to be solved)

### Key Challenges and Analysis
(Planner: Records of technical barriers, resource constraints, potential risks)

### Verifiable Success Criteria
(Planner: List measurable or verifiable goals to be achieved)

### High-level Task Breakdown
(Planner: List subtasks by phase, or break down into modules)

### Current Status / Progress Tracking
(Executor: Update completion status after each subtask. If needed, use bullet points or tables to show Done/In progress/Blocked status)

### Next Steps and Action Items
(Planner: Specific arrangements for the Executor)

### Executor's Feedback or Assistance Requests
(Executor: Write here when encountering blockers, questions, or need for more information during execution)
"""
        with open('scratchpad.md', 'w', encoding='utf-8') as f:
            f.write(initial_content)

    def _update_scratchpad_section(self, section_name: str, content: str, role: str = "Planner") -> None:
        """Update a specific section in the scratchpad.

        Args:
            section_name: Name of the section to update (without '###')
            content: New content to append to the section
            role: Role making the update ('Planner' or 'Executor')
        """
        try:
            current_content = self._get_scratchpad_content()
            sections = current_content.split('\n### ')

            # Find the target section
            target_section_idx = -1
            for i, section in enumerate(sections):
                if section.startswith(section_name) or section.startswith('### ' + section_name):
                    target_section_idx = i
                    break

            if target_section_idx == -1:
                logger.error(f"Section '{section_name}' not found in scratchpad")
                return

            # Format the new content with timestamp and role
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_content = f"\n[{role} @ {timestamp}]\n{content.strip()}\n"

            # Append the new content to the section
            if target_section_idx == 0:
                sections[0] = sections[0] + formatted_content
            else:
                sections[target_section_idx] = sections[target_section_idx] + formatted_content

            # Reconstruct the document
            updated_content = sections[0]
            for section in sections[1:]:
                updated_content += '\n### ' + section

            # Write back to file
            with open('scratchpad.md', 'w', encoding='utf-8') as f:
                f.write(updated_content)

            logger.debug(f"Updated section '{section_name}' in scratchpad")

        except Exception as e:
            logger.error(f"Error updating scratchpad section: {e}")

    def _get_scratchpad_content(self) -> str:
        """Get current content of scratchpad.md."""
        try:
            with open('scratchpad.md', 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Read scratchpad content: {content[:200]}...")
                return content
        except Exception as e:
            logger.error(f"Error reading scratchpad: {e}")
            return ""

    def _is_user_input_needed(self, response: str) -> bool:
        """Check if the response indicates need for user input."""
        # Simply check for the standardized marker
        return response.strip().startswith("WAIT_USER_CONFIRMATION")

    def _organize_output_files(self, config_path: str) -> None:
        """Move all generated files and copy the input configuration JSON file into a timestamped folder.

        Args:
            config_path: Path to the input configuration JSON file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{self.report_name}"
            os.makedirs(folder_name, exist_ok=True)
            logger.info(f"Created output folder: {folder_name}")

            # Move all created files to the folder
            for file in self.created_files:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(folder_name, os.path.basename(file)))
                    logger.debug(f"Moved {file} to {folder_name}")

            # Copy the input configuration file
            shutil.copy2(config_path, os.path.join(folder_name, os.path.basename(config_path)))
            logger.debug(f"Copied configuration file to {folder_name}")

            # Handle prompts directory
            prompts_dir = "prompts"  # Define the prompts directory path
            if self.debug and os.path.exists(prompts_dir):
                if os.path.isdir(prompts_dir):
                    # Copy the entire prompts directory
                    shutil.copytree(prompts_dir, os.path.join(folder_name, prompts_dir))
                    # Remove the original prompts directory
                    shutil.rmtree(prompts_dir)
                    logger.debug(f"Moved prompts directory to {folder_name}")
                else:
                    logger.warning(f"{prompts_dir} exists but is not a directory")
            elif self.debug:
                logger.warning(f"Prompts directory {prompts_dir} does not exist")

            logger.info(f"All files have been organized in: {folder_name}")

        except Exception as e:
            logger.error(f"Error organizing output files: {e}")

    def _get_user_feedback(self) -> bool:
        """Get user choice to continue or quit.

        Returns:
            bool: True to continue, False to quit
        """
        feedback = input("\nPress Enter to continue or 'q' to quit: ")
        return feedback.lower() != 'q'

    def chat_loop(self, initial_query: str, config_path: str) -> None:
        """Main chat loop for the research session."""
        current_query = initial_query
        conversation_history = []
        task_complete = False
        self.round_count = 0

        try:
            # Initialize Background and Motivation with the initial query
            self._update_scratchpad_section(
                "Background and Motivation",
                f"Initial research query: {initial_query}",
                "Planner"
            )

            while not task_complete:
                self.round_count += 1


                logger.info("=== Control Flow: Starting new planning cycle ===")
                # Create planner context
                planner_context = PlannerContext(
                    conversation_history=conversation_history,
                    created_files=self.created_files,
                    user_input=current_query,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.token_tracker.total_usage,
                    debug=self.debug
                )

                # Get next steps from planner
                logger.info("=== Control Flow: Requesting next steps from Planner ===")
                next_steps = self.planner.plan(planner_context)
                if not next_steps:
                    logger.error("Planner failed to provide next steps")
                    break

                logger.info(f"Planner next steps:\n{next_steps}")

                # Check if planner indicates task completion
                if next_steps.strip().startswith("TASK_COMPLETE"):
                    logger.info("Planner indicates task is complete")
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Task completed successfully - Waiting for final user feedback",
                        "Planner"
                    )

                    # Request final user feedback
                    user_input = input("\nTask completed. Please provide any additional feedback or press Enter to finish (or 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    if user_input:
                        current_query = user_input
                        self._update_scratchpad_section(
                            "Current Status / Progress Tracking",
                            f"Received additional user feedback after completion: {user_input} - Continuing task",
                            "Planner"
                        )
                        continue

                    # If no additional feedback, mark as complete and break
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Task completed and confirmed by user",
                        "Planner"
                    )
                    task_complete = True
                    break

                # If not complete, proceed with execution
                logger.info("=== Control Flow: Transferring control to Executor ===")

                # Update total usage from planner
                if planner_context.total_usage:
                    self.token_tracker.update_from_token_usage(planner_context.total_usage)

                # Create executor context
                executor_context = ExecutorContext(
                    created_files=self.created_files,
                    scratchpad_content=self._get_scratchpad_content(),
                    total_usage=self.token_tracker.total_usage,
                    debug=self.debug
                )

                # Execute the steps
                result = self.executor.execute(executor_context)
                if not result:
                    logger.error("Executor failed to provide results")
                    self._update_scratchpad_section(
                        "Executor's Feedback or Assistance Requests",
                        "Execution failed: No results provided",
                        "Executor"
                    )
                    break

                logger.info(f"Executor result:\n{result}")

                # Update total usage from executor
                if executor_context.total_usage:
                    self.token_tracker.update_from_token_usage(executor_context.total_usage)

                # Handle user input requests
                if result.strip().startswith("WAIT_USER_CONFIRMATION"):
                    self._update_scratchpad_section(
                        "Current Status / Progress Tracking",
                        "Waiting for user confirmation",
                        "Executor"
                    )
                    user_input = input("\nPlease review and provide feedback (or press Enter to continue, 'q' to quit): ")
                    if user_input.lower() == 'q':
                        break
                    if user_input:
                        current_query = user_input
                        self._update_scratchpad_section(
                            "Current Status / Progress Tracking",
                            f"Received user feedback: {user_input}",
                            "Executor"
                        )
                        continue

                # Update conversation history
                conversation_history.append({"role": "assistant", "content": result})

                # Check for errors
                if result.startswith("Error"):
                    self._update_scratchpad_section(
                        "Executor's Feedback or Assistance Requests",
                        f"Error encountered: {result}",
                        "Executor"
                    )
                    break

                logger.info("=== Control Flow: Executor completed, returning control to Planner ===")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self._update_scratchpad_section(
                "Current Status / Progress Tracking",
                "Task interrupted by user",
                "Planner"
            )
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            self._update_scratchpad_section(
                "Current Status / Progress Tracking",
                f"Error occurred: {str(e)}",
                "Planner"
            )
        finally:
            self.print_total_usage()
            # Organize output files after task completion
            self._organize_output_files(config_path)

    def print_total_usage(self) -> None:
        """Print total token usage statistics."""
        self.token_tracker.print_total_usage()

def main() -> None:
    """Main function to parse arguments and start the research session."""
    #args = parse_args()
    args = parse_args_config()

    try:
        config = load_config(args.config)

        if args.debug:
            setup_logging(args.debug)
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Create and run research session
        session = ResearchSession(
            planner_model=config['planner_model'],
            executor_model=config['executor_model'],
            report_name=config.get('report_name'),
            debug=args.debug
        )

        # Start the research with the query from config
        session.chat_loop(config['query'], args.config)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
