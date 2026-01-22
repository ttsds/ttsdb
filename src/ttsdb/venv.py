import os
import sys
import subprocess
import re
import shlex

from ttsdb.env import VENV_PATH


class Venv:
    def __init__(self, name, override=False):
        # check if name is valid directory name
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                "Name must contain only letters, numbers, underscores, and hyphens"
            )
        self.venv_path = VENV_PATH / name
        if not self.venv_path.exists() or override:
            print(f"Creating venv for {name} at {self.venv_path}")
            self.create()
        self.pip = self.venv_path / "bin" / "pip"
        self.processes = {}  # Store running processes by command

    def create(self):
        subprocess.run([sys.executable, "-m", "venv", self.venv_path], check=True)

    def install(self, requirements_path):
        subprocess.run([self.pip, "install", "-r", requirements_path], check=True)

    def run(self, command, cwd=None, env=None):
        """Run a command in the virtual environment (venv is activated).

        Args:
            command: List of command arguments (e.g., ['python', 'script.py', 'arg1'])
                     or any command available in the venv (e.g., ['pip', 'install', 'package'])
            cwd: Working directory for the command
            env: Optional dict of environment variables to set (merged with current env)

        If another command is still running, blocks until it completes.
        Otherwise, starts the command non-blocking.

        Returns a subprocess.Popen object that can be stopped externally.
        """
        # Clean up finished processes
        finished_commands = [
            cmd for cmd, proc in self.processes.items() if proc.poll() is not None
        ]
        for cmd in finished_commands:
            del self.processes[cmd]

        # Wait for any running processes to complete
        running_processes = [
            proc for proc in self.processes.values() if proc.poll() is None
        ]
        for proc in running_processes:
            proc.wait()  # Block until it completes

        # Clean up the processes we just waited for
        self.processes.clear()

        # Prepare environment variables (merge with current env)
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Now start the new process non-blocking with venv activated
        activate_script = self.venv_path / "bin" / "activate"
        # Convert command list to a shell command string with proper escaping
        cmd_str = " ".join([shlex.quote(str(arg)) for arg in command])
        # Use bash to source activate script and run the command
        bash_cmd = f"source {shlex.quote(str(activate_script))} && {cmd_str}"
        process = subprocess.Popen(["bash", "-c", bash_cmd], cwd=cwd, env=process_env)
        # Store process for potential cleanup (use tuple as key since lists aren't hashable)
        cmd_key = tuple(command)
        self.processes[cmd_key] = process
        return process

    def stop(self, command=None):
        """Stop a running process. If command is None, stops all processes.

        Args:
            command: List of command arguments to stop (e.g., ['script.py', 'arg1'])
        """
        if command is None:
            for proc in self.processes.values():
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
        else:
            # Convert command list to tuple for dictionary lookup
            cmd_key = tuple(command)
            if cmd_key in self.processes:
                proc = self.processes[cmd_key]
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
