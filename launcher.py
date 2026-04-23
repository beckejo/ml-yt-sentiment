#!/usr/bin/env python
"""
Unified launcher for ML sentiment analysis project.
Starts training, API server, Streamlit app, and MLflow UI in one command.
"""

import subprocess
import sys
import time
import atexit
import os
import socket
from pathlib import Path
from typing import List, Optional, Dict, Any

# Color codes for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

services: List[Dict[str, Any]] = []


def cleanup():
    """Kill all child processes on exit."""
    print(f"\n{YELLOW}Cleaning up processes...{RESET}")
    for svc in services:
        proc = svc.get("process")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
        log_handle = svc.get("log_handle")
        if log_handle and not log_handle.closed:
            try:
                log_handle.close()
            except Exception:
                pass
    print(f"{GREEN}Cleanup complete.{RESET}")


def _print_log_tail(log_path: Path, lines: int = 12):
    """Print tail lines from a service log file."""
    try:
        if not log_path.exists():
            return
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = content[-lines:]
        if tail:
            print("  Last log lines:")
            for line in tail:
                print(f"    {line}")
    except Exception:
        return


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP port is already bound/listening on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _pids_listening_on_port(port: int) -> list[int]:
    """Best-effort process id discovery for a listening TCP port."""
    pids: set[int] = set()
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    port_token = f":{port}"
    for line in output.splitlines():
        upper = line.upper()
        if "LISTENING" not in upper or port_token not in line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        local_addr = parts[1]
        state = parts[3].upper()
        pid_str = parts[4]
        if state != "LISTENING" or not local_addr.endswith(port_token):
            continue

        try:
            pids.add(int(pid_str))
        except ValueError:
            continue

    return sorted(pids)


def stop_services_on_ports(ports: list[int]) -> int:
    """Stop any listening processes on known service ports."""
    print(f"\n{BLUE}Stopping services on configured ports...{RESET}")
    stopped_any = False
    for port in ports:
        pids = _pids_listening_on_port(port)
        if not pids:
            print(f"{YELLOW}Port {port}: no listening process found.{RESET}")
            continue

        for pid in pids:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                stopped_any = True
                print(f"{GREEN}Port {port}: stopped PID {pid}.{RESET}")
            except Exception as exc:
                print(f"{RED}Port {port}: failed to stop PID {pid}: {exc}{RESET}")

    if stopped_any:
        print(f"{GREEN}Service stop operation completed.{RESET}")
        return 0

    print(f"{YELLOW}No running services were found to stop.{RESET}")
    return 0


def run_training(data_dir: Optional[str] = None):
    """Run the training pipeline."""
    print(f"\n{BLUE}Starting training pipeline...{RESET}")
    
    # Build environment with optional data directory
    env = os.environ.copy()
    if data_dir:
        # Set data path if provided
        env["DATA_DIR"] = str(data_dir)
    
    # Ensure hand-labeled path is valid. If not, fallback to local Reddit CSV.
    configured_eval_path = env.get("HAND_LABELED_TEST_PATH", "").strip()
    project_dir = Path(__file__).parent
    fallback_candidates = [
        project_dir / "data" / "reddit_local.csv",
        project_dir / "reddit_local.csv",
    ]
    fallback_eval_path = next((str(path) for path in fallback_candidates if path.exists()), "")

    configured_is_placeholder = configured_eval_path.lower().startswith("path\\to\\")
    configured_missing = not configured_eval_path
    configured_invalid = configured_eval_path and not Path(configured_eval_path).exists()

    if configured_missing or configured_is_placeholder or configured_invalid:
        if fallback_eval_path:
            env["HAND_LABELED_TEST_PATH"] = fallback_eval_path
            print(
                f"{YELLOW}HAND_LABELED_TEST_PATH was missing/invalid. "
                f"Using fallback dataset: {fallback_eval_path}{RESET}"
            )
        else:
            print(
                f"{YELLOW}No valid HAND_LABELED_TEST_PATH and no local fallback found. "
                f"Set it with: $env:HAND_LABELED_TEST_PATH='C:\\path\\to\\hand_labeled.csv'{RESET}"
            )
    
    try:
        result = subprocess.run(
            [sys.executable, "models.py"],
            env=env,
            cwd=Path(__file__).parent,
            capture_output=False
        )
        if result.returncode == 0:
            print(f"{GREEN}Training completed successfully.{RESET}")
            return True
        else:
            print(f"{RED}Training failed with exit code {result.returncode}.{RESET}")
            return False
    except Exception as e:
        print(f"{RED}Error running training: {e}{RESET}")
        return False


def start_process(
    cmd: List[str],
    name: str,
    port: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> bool:
    """Start a subprocess and add to cleanup list."""
    try:
        print(f"{BLUE}Starting {name}...{RESET}", end=" ")
        proc_env = os.environ.copy()
        if extra_env:
            proc_env.update(extra_env)

        logs_dir = Path(__file__).parent / "logs" / "launcher"
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_")
        log_path = logs_dir / f"{safe_name}.log"
        log_handle = open(log_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            env=proc_env,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            bufsize=1
        )
        services.append(
            {
                "name": name,
                "port": port,
                "process": proc,
                "reported_stop": False,
                "log_path": log_path,
                "log_handle": log_handle,
            }
        )
        time.sleep(2)  # Give it time to start
        
        if proc.poll() is not None:
            # Process already exited (error)
            print(f"{RED}FAILED{RESET}")
            print(f"  Exit code: {proc.returncode}")
            print(f"  See log: {log_path}")
            _print_log_tail(log_path)
            return False
        else:
            port_info = f" (port {port})" if port else ""
            print(f"{GREEN}OK{RESET}{port_info}")
            print(f"  Log file: {log_path}")
            return True
    except Exception as e:
        print(f"{RED}FAILED: {e}{RESET}")
        return False


def main():
    """Main launcher orchestration."""
    print(f"\n{BLUE}{'='*60}")
    print("ML Sentiment Analysis - Unified Launcher")
    print(f"{'='*60}{RESET}\n")
    
    # Register cleanup handler
    atexit.register(cleanup)
    
    # Parse arguments
    skip_training = "--skip-training" in sys.argv
    skip_api = "--skip-api" in sys.argv
    skip_streamlit = "--skip-streamlit" in sys.argv
    skip_mlflow = "--skip-mlflow" in sys.argv
    stop_services = "--stop-services" in sys.argv
    api_port = 8001
    streamlit_port = 8501
    mlflow_port = 5000
    
    # Show usage
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python launcher.py [OPTIONS]")
        print("\nOptions:")
        print("  --skip-training      Skip model training")
        print("  --skip-api           Skip FastAPI server")
        print("  --skip-streamlit     Skip Streamlit app")
        print("  --skip-mlflow        Skip MLflow UI")
        print("  --stop-services      Stop services on default ports and exit")
        print("  --help               Show this help message")
        print("\nDefault: Runs training, then starts API, Streamlit, and MLflow UI")
        return 0

    if stop_services:
        return stop_services_on_ports([mlflow_port, api_port, streamlit_port])
    
    # Step 1: Training
    if not skip_training:
        if not run_training():
            print(f"{YELLOW}Continuing despite training failure...{RESET}")
            # Don't exit - let user try to use existing champion
    else:
        print(f"{YELLOW}Skipping training (--skip-training).{RESET}")
    
    # Step 2: Start services
    print(f"\n{BLUE}{'='*60}")
    print("Starting services...")
    print(f"{'='*60}{RESET}\n")
    
    services_started = 0
    
    # Start MLflow UI
    if not skip_mlflow:
        if is_port_in_use(mlflow_port):
            print(
                f"{YELLOW}MLflow UI port {mlflow_port} is already in use. "
                f"Reusing existing service.{RESET}"
            )
            services.append(
                {
                    "name": "MLflow UI",
                    "port": mlflow_port,
                    "process": None,
                    "reported_stop": False,
                    "reused": True,
                }
            )
            services_started += 1
        elif start_process(
            [sys.executable, "-m", "mlflow", "ui",
             "--backend-store-uri", "sqlite:///mlflow.db",
             "--host", "127.0.0.1",
             "--port", str(mlflow_port)],
            "MLflow UI",
            mlflow_port
        ):
            services_started += 1
    
    # Start FastAPI
    if not skip_api:
        if is_port_in_use(api_port):
            print(
                f"{YELLOW}FastAPI port {api_port} is already in use. "
                f"Reusing existing service.{RESET}"
            )
            services.append(
                {
                    "name": "FastAPI Server",
                    "port": api_port,
                    "process": None,
                    "reported_stop": False,
                    "reused": True,
                }
            )
            services_started += 1
        elif start_process(
            [sys.executable, "-m", "uvicorn",
             "fastapi_app:app",
             "--host", "127.0.0.1",
             "--port", str(api_port)],
            "FastAPI Server",
            api_port
        ):
            services_started += 1
    
    # Start Streamlit
    if not skip_streamlit:
        if is_port_in_use(streamlit_port):
            print(
                f"{YELLOW}Streamlit port {streamlit_port} is already in use. "
                f"Reusing existing service.{RESET}"
            )
            services.append(
                {
                    "name": "Streamlit App",
                    "port": streamlit_port,
                    "process": None,
                    "reported_stop": False,
                    "reused": True,
                }
            )
            services_started += 1
        elif start_process(
            [sys.executable, "-m", "streamlit", "run",
             "streamlit_app.py",
             "--server.port", str(streamlit_port),
             "--server.address", "127.0.0.1"],
            "Streamlit App",
            streamlit_port,
            extra_env={"FASTAPI_BASE_URL": f"http://127.0.0.1:{api_port}"},
        ):
            services_started += 1
    
    # Summary
    print(f"\n{BLUE}{'='*60}")
    print("Services Status")
    print(f"{'='*60}{RESET}")
    
    if services_started > 0:
        print(f"\n{GREEN}✓ Successfully started {services_started} service(s):{RESET}")
        if not skip_mlflow:
            print(f"  • MLflow UI:      http://127.0.0.1:{mlflow_port}")
        if not skip_api:
            print(f"  • FastAPI:        http://127.0.0.1:{api_port}/docs")
        if not skip_streamlit:
            print(f"  • Streamlit:      http://127.0.0.1:{streamlit_port}")
        managed_count = sum(1 for svc in services if svc.get("process") is not None)
        reused_count = sum(1 for svc in services if svc.get("process") is None)

        if managed_count > 0 and reused_count == 0:
            print(f"\n{YELLOW}Press Ctrl+C to stop all services.{RESET}\n")
        elif managed_count > 0 and reused_count > 0:
            print(
                f"\n{YELLOW}Press Ctrl+C to stop launcher-started services. "
                f"Reused services will keep running.{RESET}\n"
            )
        else:
            print(
                f"\n{YELLOW}Press Ctrl+C to exit monitoring. "
                f"Reused services will keep running.{RESET}\n"
            )
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                # Check if any process has died
                alive_count = 0
                for svc in services:
                    proc = svc.get("process")
                    if proc is None:
                        port = svc.get("port")
                        if isinstance(port, int) and is_port_in_use(port):
                            alive_count += 1
                            continue

                        if not svc.get("reported_stop", False):
                            svc["reported_stop"] = True
                            name = svc.get("name", "Unknown service")
                            print(f"{RED}Warning: {name} is no longer listening on port {port}.{RESET}")
                        continue

                    if proc.poll() is None:
                        alive_count += 1
                        continue

                    if not svc.get("reported_stop", False):
                        svc["reported_stop"] = True
                        name = svc.get("name", "Unknown service")
                        code = proc.returncode
                        log_path = svc.get("log_path")
                        print(f"{RED}Warning: {name} has stopped (exit code {code}).{RESET}")
                        if log_path:
                            print(f"  See log: {log_path}")
                            _print_log_tail(log_path)

                if alive_count == 0:
                    print(f"{RED}All services have stopped. Exiting launcher.{RESET}")
                    break
        except KeyboardInterrupt:
            pass
    else:
        print(f"{RED}No services started. Check configuration and try again.{RESET}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
