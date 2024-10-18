import pathlib
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import giturlparse
from bs4 import BeautifulSoup, Comment
from git import Repo

from bot338.utils import get_logger

logger = get_logger(__name__)


def get_git_command(id_file: Path) -> str:
    """Get the git command with the given id file.

    Args:
        id_file: The path to the id file.

    Returns:
        The git command with the id file.
    """
    assert id_file.is_file()

    git_command = f"ssh -v -i /{id_file}"
    return git_command


def fetch_git_remote_hash(
    repo_url: Optional[str] = None, id_file: Optional[Path] = None
) -> Optional[str]:
    """Fetch the remote hash of the git repository.

    Args:
        repo_url: The URL of the git repository.
        id_file: The path to the id file.

    Returns:
        The remote hash of the git repository.
    """
    if repo_url is None:
        logger.warning(f"No repo url was supplied. Not returning a repo hash")
        return None
    git_command = get_git_command(id_file)
    repo_url = giturlparse.parse(repo_url).urls.get("ssh")

    cmd = f'GIT_SSH_COMMAND="{git_command} -o IdentitiesOnly=yes" git ls-remote {repo_url}'
    normal = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    sha = re.split(r"\t+", normal.stdout)[0]
    return sha


def fetch_repo_metadata(repo: "Repo") -> Dict[str, str]:
    """Fetch the metadata of the git repository.

    Args:
        repo: The git repository.

    Returns:
        The metadata of the git repository.
    """
    head_commit = repo.head.commit

    return dict(
        commit_summary=head_commit.summary,
        commit_message=head_commit.message,
        commit_author=str(head_commit.author),
        commit_time=head_commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        commit_hash=head_commit.hexsha,
        commit_stats=head_commit.stats.total,
    )


def fetch_git_repo(paths: Any, id_file: Path) -> Dict[str, str]:
    """Fetch the git repository.

    Args:
        paths: The paths of the git repository.
        id_file: The path to the id file.

    Returns:
        The metadata of the git repository.
    """
    git_command = get_git_command(id_file)

    if paths.local_path.is_dir():
        repo = Repo(paths.local_path)
        logger.debug(
            f"Repo {paths.local_path} already exists... Pulling changes from {repo.remotes.origin.url}"
        )
        with repo.git.custom_environment(GIT_SSH_COMMAND=git_command):
            if paths.branch is not None:
                repo.git.checkout(paths.branch)
            repo.remotes.origin.pull()
    else:
        remote_url = giturlparse.parse(f"{paths.repo_path}").urls.get("ssh")

        logger.debug(f"Cloning {remote_url} to {paths.local_path}")
        repo = Repo.clone_from(
            remote_url, paths.local_path, env=dict(GIT_SSH_COMMAND=git_command)
        )
        if paths.branch is not None:
            repo.git.checkout(paths.branch)
    return fetch_repo_metadata(repo)
