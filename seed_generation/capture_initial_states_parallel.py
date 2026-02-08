"""
Parallel version of initial state capture using multiple AWS instances.

This script captures initial screenshots, accessibility trees, and Set-of-Marks (SoM)
tagged screenshots for OSWorld tasks using multiple AWS EC2 instances in parallel
for faster processing.

Usage:
    # Capture OS domain with 5 parallel instances
    python capture_initial_states_parallel.py \
        --domain os \
        --provider_name aws \
        --region us-east-1 \
        --num_envs 5 \
        --output_dir ./initial_states

    # Capture all domains with 10 parallel instances
    python capture_initial_states_parallel.py \
        --domain os \
        --provider_name aws \
        --num_envs 10 \
        --output_dir ./initial_states

Outputs per task:
    - initial_screenshot.png: Original screenshot
    - initial_a11y_tree.json: Accessibility tree XML
    - initial_som_screenshot.png: Screenshot with numbered bounding boxes
    - initial_som_elements.txt: Table mapping numbers to UI elements
    - initial_som_marks.json: Bounding box coordinates
    - metadata.json: Task metadata and capture information

Note: For domains with slow-loading applications (vscode, libreoffice_*, chrome),
use --post_reset_delay to wait for apps to fully load before capturing.
"""

import argparse
import json
import logging
import os
import sys
import glob
import time
import signal
from datetime import datetime
from multiprocessing import Process, Manager, current_process
from typing import Dict, List, Any, Tuple
from PIL import Image
import io
import xml.etree.ElementTree as ET

# Add parent directory to Python path to import OSWorld modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from desktop_env.desktop_env import DesktopEnv
from mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for signal handling
processes = []
is_terminating = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("capture_parallel")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel capture of initial states for OSWorld tasks"
    )

    # Task selection
    parser.add_argument(
        "--domain",
        type=str,
        default="os",
        help="Domain to capture (os, chrome, etc.) or 'all'"
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Specific task IDs (space-separated)"
    )
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="../evaluation_examples/test_all.json",
        help="Path to test_all.json (default: ../evaluation_examples/test_all.json)"
    )
    parser.add_argument(
        "--examples_base_dir",
        type=str,
        default="../evaluation_examples/examples",
        help="Base directory with task JSON files (default: ../evaluation_examples/examples)"
    )

    # Environment configuration
    parser.add_argument(
        "--provider_name",
        type=str,
        default="aws",
        choices=["aws", "vmware", "virtualbox", "docker"],
        help="VM provider (aws recommended for parallel)"
    )
    parser.add_argument(
        "--path_to_vm",
        type=str,
        default=None,
        help="Path to VM (for vmware/virtualbox)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--os_type",
        type=str,
        default="Ubuntu"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (default: True)"
    )
    parser.add_argument(
        "--client_password",
        type=str,
        default=""
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=1920
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=1080
    )

    # Parallel execution
    parser.add_argument(
        "--num_envs",
        type=int,
        default=5,
        help="Number of parallel environments (AWS instances)"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./initial_states",
        help="Output directory for captured states (default: ./initial_states/)"
    )
    parser.add_argument(
        "--capture_screenshot",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--capture_a11y_tree",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--capture_som",
        action="store_true",
        default=True,
        help="Capture Set-of-Marks (SoM) tagged screenshot"
    )
    parser.add_argument(
        "--post_reset_delay",
        type=float,
        default=60.0,
        help="Seconds to wait after env.reset() before capturing (for slow apps like VSCode)"
    )

    parser.add_argument("--aws_ami", type=str, default=None)
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--config_type", choices=["redteamcua", "osworld"], default="redteamcua", help="Defines the config type used by an example (currently hacky)")


    return parser.parse_args()


def load_task_list(args) -> List[Tuple[str, str, str]]:
    """
    Load task list and return as [(domain, task_id, task_file_path), ...].
    """
    task_list = []

    # If specific task IDs provided
    if args.task_ids:
        logger.info(f"Loading {len(args.task_ids)} specific tasks...")
        for task_id in args.task_ids:
            pattern = f"{args.examples_base_dir}/**/{task_id}*.json"
            matches = glob.glob(pattern, recursive=True)
            if matches:
                task_file = matches[0]
                domain = os.path.basename(os.path.dirname(task_file))
                task_list.append((domain, task_id, task_file))
                logger.info(f"  Found: {task_id} in {domain}")
            else:
                logger.warning(f"  Not found: {task_id}")
        return task_list

    # Load from test_all.json
    with open(args.test_all_meta_path, 'r') as f:
        test_all = json.load(f)

    # Determine which domains to process
    if args.domain == "all":
        domains = test_all.keys()
    else:
        if args.domain not in test_all:
            logger.error(f"Domain '{args.domain}' not found")
            sys.exit(1)
        domains = [args.domain]

    logger.info(f"Loading tasks from domains: {list(domains)}")

    # Build task list
    for domain in domains:
        task_ids = test_all[domain]
        logger.info(f"  Domain '{domain}': {len(task_ids)} tasks")

        for task_id in task_ids:
            task_file = os.path.join(args.examples_base_dir, domain, f"{task_id}.json")
            if os.path.exists(task_file):
                task_list.append((domain, task_id, task_file))
            else:
                logger.warning(f"    Not found: {task_file}")

    logger.info(f"Total tasks loaded: {len(task_list)}")
    return task_list


def save_screenshot(screenshot_bytes: bytes, output_path: str):
    """Save screenshot to PNG."""
    try:
        image = Image.open(io.BytesIO(screenshot_bytes))
        image.save(output_path, 'PNG')
        return True
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return False


def save_a11y_tree(a11y_tree: Any, output_path: str):
    """Save accessibility tree to JSON."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(a11y_tree, str):
                f.write(a11y_tree)
            else:
                json.dump(a11y_tree, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save a11y tree: {e}")
        return False


def save_som_screenshot(screenshot_bytes: bytes, a11y_tree_xml: str, platform: str, output_dir: str):
    """
    Generate and save Set-of-Marks (SoM) tagged screenshot and element list.

    Args:
        screenshot_bytes: Original screenshot bytes
        a11y_tree_xml: Accessibility tree XML string
        platform: OS platform (ubuntu or windows)
        output_dir: Directory to save SoM files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Parse accessibility tree and filter nodes
        root = ET.fromstring(a11y_tree_xml)
        nodes = filter_nodes(root, platform=platform, check_image=True)

        # Draw bounding boxes and generate tagged screenshot
        marks, drew_nodes, element_list, tagged_screenshot_bytes = draw_bounding_boxes(
            nodes, screenshot_bytes, down_sampling_ratio=1.0, platform=platform
        )

        # Save tagged screenshot
        som_screenshot_path = os.path.join(output_dir, "initial_som_screenshot.png")
        image = Image.open(io.BytesIO(tagged_screenshot_bytes))
        image.save(som_screenshot_path, 'PNG')

        # Save element list (TSV format: index, tag, name, text)
        som_elements_path = os.path.join(output_dir, "initial_som_elements.txt")
        with open(som_elements_path, 'w', encoding='utf-8') as f:
            f.write(element_list)

        # Save marks (bounding box coordinates)
        som_marks_path = os.path.join(output_dir, "initial_som_marks.json")
        with open(som_marks_path, 'w', encoding='utf-8') as f:
            json.dump(marks, f, indent=2)

        logger.info(f"    Saved SoM screenshot, {len(marks)} elements marked")
        return True

    except Exception as e:
        logger.error(f"Failed to save SoM screenshot: {e}")
        return False


def save_metadata(task_config: Dict, observation: Dict, output_path: str, has_som: bool = False):
    """Save task metadata."""
    try:
        metadata = {
            "task_id": task_config.get("id"),
            "domain": task_config.get("_domain", "unknown"),
            "instruction": task_config.get("instruction"),
            "snapshot": task_config.get("snapshot", "default"),
            "config_steps": len(task_config.get("config", [])),
            "has_screenshot": observation.get("screenshot") is not None,
            "has_a11y_tree": observation.get("accessibility_tree") is not None,
            "has_som": has_som,
            "capture_timestamp": datetime.now().isoformat(),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return False


def capture_worker(task_queue, args, shared_results):
    """
    Worker process that captures initial states for tasks from the queue.
    Each worker creates its own AWS instance.
    """
    worker_name = current_process().name
    env = None

    try:
        # Initialize environment for this worker
        logger.info(f"[{worker_name}] Initializing environment...")

        # Get AMI ID for AWS
        if args.provider_name == "aws":
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP
            #OSWorld AMI ID
            screen_size = (args.screen_width, args.screen_height)
            ami_id = IMAGE_ID_MAP[args.region].get(screen_size, IMAGE_ID_MAP[args.region][(1920, 1080)])
        else:
            ami_id = None

        env = DesktopEnv(
            provider_name=args.provider_name,
            path_to_vm=args.path_to_vm,
            region=args.region,
            snapshot_name=ami_id,
            os_type=args.os_type,
            headless=args.headless,
            action_space="pyautogui",
            require_a11y_tree=args.capture_a11y_tree,
            require_terminal=False,
            screen_size=(args.screen_width, args.screen_height),
            client_password=args.client_password
        )
        
        logger.info(f"[{worker_name}] Environment initialized!")

        # Process tasks from queue
        while True:
            try:
                # Get task from queue with timeout
                task_item = task_queue.get(timeout=5)
            except:
                # Queue is empty, exit
                break

            domain, task_id, task_file = task_item

            try:
                logger.info(f"[{worker_name}] Processing: {domain}/{task_id}")

                # Load task config
                with open(task_file, 'r') as f:
                    task_config = json.load(f)
                    task_config['_domain'] = domain

                # Create output directory
                task_output_dir = os.path.join(args.output_dir, domain, task_id)
                os.makedirs(task_output_dir, exist_ok=True)

                # Reset environment
                logger.info(f"[{worker_name}] Resetting environment...")
                observation = env.reset(task_config=task_config)
                # env.prepare_injection(task_config=task_config, config_type=args.config_type)
                observation = env._get_obs() # Get the initial observation
                logger.info(f"[{worker_name}] Reset complete!")

                # Wait for applications to fully load (important for slow apps like VSCode, LibreOffice)
                if args.post_reset_delay > 0:
                    logger.info(f"[{worker_name}] Waiting {args.post_reset_delay}s for apps to load...")
                    time.sleep(args.post_reset_delay)
                    # Get fresh observation after delay
                    observation = env._get_obs()
                    logger.info(f"[{worker_name}] Post-delay observation captured")

                # Save screenshot
                if args.capture_screenshot and observation.get("screenshot"):
                    screenshot_path = os.path.join(task_output_dir, "initial_screenshot.png")
                    if save_screenshot(observation["screenshot"], screenshot_path):
                        logger.info(f"[{worker_name}] Saved screenshot")

                # Save a11y tree
                if args.capture_a11y_tree and observation.get("accessibility_tree"):
                    a11y_path = os.path.join(task_output_dir, "initial_a11y_tree.json")
                    if save_a11y_tree(observation["accessibility_tree"], a11y_path):
                        logger.info(f"[{worker_name}] Saved a11y tree")

                # Save SoM (Set-of-Marks) screenshot
                has_som = False
                if args.capture_som and observation.get("screenshot") and observation.get("accessibility_tree"):
                    try:
                        logger.info(f"[{worker_name}] Generating SoM screenshot...")
                        has_som = save_som_screenshot(
                            observation["screenshot"],
                            observation["accessibility_tree"],
                            platform=args.os_type.lower(),
                            output_dir=task_output_dir
                        )
                        if has_som:
                            logger.info(f"[{worker_name}] Saved SoM screenshot")
                    except Exception as e:
                        logger.error(f"[{worker_name}] Failed to generate SoM: {e}")

                # Save metadata
                metadata_path = os.path.join(task_output_dir, "metadata.json")
                save_metadata(task_config, observation, metadata_path, has_som=has_som)

                # Record success
                shared_results.append({
                    "task_id": task_id,
                    "domain": domain,
                    "status": "success",
                    "worker": worker_name
                })

                logger.info(f"[{worker_name}] ✓ Completed: {domain}/{task_id}")

            except Exception as e:
                logger.error(f"[{worker_name}] ✗ Failed {domain}/{task_id}: {e}", exc_info=True)
                shared_results.append({
                    "task_id": task_id,
                    "domain": domain,
                    "status": "failed",
                    "error": str(e),
                    "worker": worker_name
                })
                continue

    except Exception as e:
        logger.error(f"[{worker_name}] Worker error: {e}", exc_info=True)

    finally:
        # Cleanup environment
        logger.info(f"[{worker_name}] Cleaning up...")
        if env:
            try:
                env.close()
                env.terminate()
                logger.info(f"[{worker_name}] Environment closed")
            except Exception as e:
                logger.error(f"[{worker_name}] Cleanup error: {e}")


def signal_handler(signum, frame):
    """Handle Ctrl+C and termination signals."""
    global is_terminating, processes

    if is_terminating:
        return

    is_terminating = True
    logger.info(f"Received signal {signum}. Shutting down...")

    # Terminate all worker processes
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Terminating process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")

    # Wait for processes to terminate
    time.sleep(2)

    # Force kill if needed
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Force killing process {p.name}...")
                os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error force killing: {e}")

    logger.info("Shutdown complete")
    sys.exit(0)


def main():
    global processes

    args = parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("="*80)
    logger.info("OSWorld Parallel Initial State Capture")
    logger.info("="*80)
    logger.info(f"Domain:             {args.domain}")
    logger.info(f"Provider:           {args.provider_name}")
    logger.info(f"Num Envs:           {args.num_envs}")
    logger.info(f"Region:             {args.region}")
    logger.info(f"Output dir:         {args.output_dir}")
    logger.info(f"Screenshot:         {args.capture_screenshot}")
    logger.info(f"A11y tree:          {args.capture_a11y_tree}")
    logger.info(f"SoM (Set-of-Marks): {args.capture_som}")
    logger.info(f"Post-reset delay:   {args.post_reset_delay}s")
    logger.info("="*80)

    # SoM requires a11y tree
    if args.capture_som and not args.capture_a11y_tree:
        logger.warning("SoM capture requires a11y tree. Enabling a11y tree capture.")
        args.capture_a11y_tree = True

    # Load task list
    task_list = load_task_list(args)
    if not task_list:
        logger.error("No tasks to process!")
        return

    logger.info(f"\nTotal tasks to capture: {len(task_list)}")

    # Setup multiprocessing
    with Manager() as manager:
        shared_results = manager.list()
        task_queue = manager.Queue()

        # Fill task queue
        for task_item in task_list:
            task_queue.put(task_item)

        # Start worker processes
        logger.info(f"\nStarting {args.num_envs} worker processes...")
        processes = []

        for i in range(args.num_envs):
            p = Process(
                target=capture_worker,
                args=(task_queue, args, shared_results),
                name=f"CaptureWorker-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started {p.name} (PID: {p.pid})")

        try:
            # Monitor processes
            while True:
                alive_count = sum(1 for p in processes if p.is_alive())

                if task_queue.empty() and alive_count == 0:
                    logger.info("\nAll tasks completed!")
                    break

                if alive_count == 0 and not task_queue.empty():
                    logger.error("\nAll workers died but tasks remain!")
                    break

                # Show progress
                completed = len([r for r in shared_results if r])
                logger.info(f"Progress: {completed}/{len(task_list)} tasks completed, {alive_count} workers active")

                time.sleep(10)

            # Wait for all processes to finish
            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            raise

        # Gather results
        results = list(shared_results)

    # Print summary
    successful = len([r for r in results if r.get("status") == "success"])
    failed = len([r for r in results if r.get("status") == "failed"])

    logger.info("\n" + "="*80)
    logger.info("CAPTURE SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tasks:      {len(task_list)}")
    logger.info(f"Successful:       {successful}")
    logger.info(f"Failed:           {failed}")
    logger.info(f"Success rate:     {successful/len(task_list)*100:.1f}%")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info("="*80)

    # Save detailed results
    results_file = os.path.join(args.output_dir, "capture_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
