#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapted from github.com/taichi-dev/taichi (Apache 2.0 licensed)
# Source: https://github.com/taichi-dev/taichi/blob/master/.github/workflows/scripts/ghstack-perm-check.py

import argparse
import json
import os
import re
import subprocess
import sys

import requests


def get_session(token):
    gh = requests.Session()
    gh.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return gh


# Check condition. Post GitHub comment and exit(1) in false
def must(session, repo, pr, cond, msg):
    if not cond:
        print(msg)
        session.post(
            f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
            json={
                "body": f"fms bot failed: {msg}",
            },
        )
        exit(1)


# Check if the PR is in GHStack format or GitHub format
def is_ghstack(session, event):
    # Extract the PR from the event
    pr = event["event"]["client_payload"]["pull_request"]
    must(session, event["repository"], 0, pr, "No PR object found in the event")

    # Check the head_ref
    head_ref = pr["head"]["ref"]
    must(
        session,
        event["repository"],
        pr["number"],
        head_ref,
        "Not head ref found in the event",
    )
    return bool(re.match(r"^gh/[A-Za-z0-9-]+/[0-9]+/head$", head_ref))


class ChatOps:
    def __init__(self, session, event):
        self.gh = session
        self.event = event

        # Extract some useful data from the event
        self.PR = event["event"]["client_payload"]["pull_request"]
        self.REPO = event["repository"]
        self.NUMBER = self.PR["number"]
        self.PR_URL = self.PR["html_url"]
        self.PR_FROM_FORK = self.PR["head"]["repo"]["fork"]
        self.pr_ref = self.get_pr_ref()

        # Fetch the required branches locally
        print(":: Fetching newest main...")
        self.must(os.system("git fetch origin main") == 0, "Can't fetch main")
        print(":: Fetching orig branch...")
        self.must(
            os.system(f"git fetch origin {self.pr_ref}") == 0,
            "Can't fetch orig branch {self.pr_ref}",
        )

    def must(self, cond, msg):
        must(self.gh, self.REPO, self.NUMBER, cond, msg)

    # Prerequisites before a PR can land
    # If no pr is specified, self.NUMBER is used
    def land_prerequisites(self, pr=None):
        if pr is None:
            pr = self.NUMBER
        # Checking Approvals
        print(f":: Checking PR status #{pr}... ", end="")
        resp = self.gh.get(f"https://api.github.com/repos/{self.REPO}/pulls/{pr}")
        self.must(resp.ok, "Error Getting PR Object!")
        pr_obj = resp.json()

        resp = self.gh.get(
            f"https://api.github.com/repos/{self.REPO}/pulls/{pr}/reviews"
        )
        self.must(resp.ok, "Error Getting PR Reviews!")
        reviews = resp.json()
        idmap = {}
        approved = False
        for r in reviews:
            s = r["state"]
            if s not in ("COMMENTED",):
                idmap[r["user"]["login"]] = r["state"]

        for u, cc in idmap.items():
            approved = approved or cc == "APPROVED"
            self.must(
                cc in ("APPROVED", "DISMISSED"),
                f"@{pr} has stamped PR #{pr} `{cc}`, please resolve it first!",
            )

        self.must(approved, f"PR #{pr} is not approved yet!")

        # Checking CI Jobs
        resp = self.gh.get(
            f'https://api.github.com/repos/{self.REPO}/commits/{pr_obj["head"]["sha"]}/check-runs'
        )
        self.must(resp.ok, "Error getting check runs status!")
        checkruns = resp.json()
        for cr in checkruns["check_runs"]:
            status = cr.get("conclusion", cr["status"])
            name = cr["name"]
            if name == "Copilot for PRs":
                continue
            self.must(
                status in ("success", "neutral"),
                f"PR #{pr} check-run `{name}`'s status `{status}` is not success!",
            )
        print("SUCCESS!")

    # Prerequisites before a PR can be rebased
    def rebase_prerequisites(self):
        # Check if the PR owner matches the actor
        actor = self.event["event"]["client_payload"]["github"]["actor"]
        self.must(actor, "Event actor not found!")

        pr_owner = self.PR["user"]["login"]
        self.must(actor, "PR Owner not found!")

        # If the comment was not left by the owner, it's only allowed for owners
        if actor != pr_owner:
            # Get the user permissions on the repo
            resp = self.gh.get(
                f"https://api.github.com/repos/{self.REPO}/collaborators/{actor}/permission"
            )
            self.must(
                resp.ok,
                f"User {actor} must be a maintainer {self.REPO} to rebase someone else's PR",
            )
            permissions = resp.json()

            # Check if the actor is an OWNER (permission "write" or "admin")
            # Reference: https://docs.github.com/en/rest/collaborators/collaborators?apiVersion=2022-11-28#check-if-a-user-is-a-repository-collaborator  # noqa: E501
            self.must(
                permissions["permission"] in ["write", "admin"],
                f"User {actor} must be a maintainer {self.REPO} to rebase someone else's PR",
            )


class GHStack(ChatOps):
    def __init__(self, session, event):
        super().__init__(session, event)
        self.pr_numbers = self.ghstack_pr_numbers()

    def get_pr_ref(self):
        return self.PR["head"]["ref"].replace("/head", "/orig")

    def ghstack_pr_numbers(self):
        proc = subprocess.Popen(
            "git log FETCH_HEAD...$(git merge-base FETCH_HEAD origin/main)",
            stdout=subprocess.PIPE,
            shell=True,
        )
        out, _ = proc.communicate()
        self.must(proc.wait() == 0, "`git log` command failed!")

        pr_numbers = re.findall(
            r"Pull Request resolved: https://github.com/.*?/pull/([0-9]+)",
            out.decode("utf-8"),
        )
        pr_numbers = list(map(int, pr_numbers))
        self.must(
            pr_numbers and pr_numbers[0] == self.NUMBER,
            "Extracted PR numbers not seems right!",
        )
        return pr_numbers

    def land(self):
        # Enforce requirements for every PR in the stack
        for n in self.pr_numbers:
            # Checking Approvals
            self.land_prerequisites(n)

        print(":: All PRs are ready to be landed!")

    def rebase(self):
        self.rebase_prerequisites()
        print(":: All PRs are ready to be rebased!")


class GitHub(ChatOps):
    def get_pr_ref(self):
        return f"pull/{self.NUMBER}/head"

    def must(self, cond, msg):
        must(self.gh, self.REPO, self.NUMBER, cond, msg)

    def land(self):
        # Enforce requirements for the PR
        self.land_prerequisites()
        print(":: The PR is ready to be landed!")

    def rebase(self):
        self.rebase_prerequisites()
        print(":: The PR is ready to be rebased!")


def main(check):
    # Setup the wrapper class
    EV = json.loads(sys.stdin.read())
    session = get_session(os.environ["GITHUB_TOKEN"])
    chatops = None
    isgh = is_ghstack(session, EV)

    if isgh:
        chatops = GHStack(session, EV)
    else:
        chatops = GitHub(session, EV)

    # Set a number of useful environment variables
    env_file = os.getenv("GITHUB_ENV")
    with open(env_file, "a") as envfile:
        envfile.write(f"IS_GHSTACK={isgh}\n")
        envfile.write(f"REPO={chatops.REPO}\n")
        envfile.write(f"PR_NUMBER={chatops.NUMBER}\n")
        envfile.write(f"PR_REF={chatops.pr_ref}\n")
        envfile.write(f"PR_URL={chatops.PR_URL}\n")

    # Run the github action specific checks
    if check == "land":
        chatops.land()
    elif check == "rebase":
        chatops.rebase()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHStack Permission Checks")
    parser.add_argument("check", choices=["land", "rebase"])
    args = parser.parse_args()

    main(args.check)
