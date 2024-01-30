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


class GHStackChecks:
    def __init__(self, token, event):
        self.gh = requests.Session()
        self.gh.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.event = event
        self.REPO = event["repository"]
        self.PR = event["event"]["client_payload"]["pull_request"]
        self.NUMBER = self.PR["number"]
        self.pr_numbers = self.shared_checks()

    def must(self, cond, msg):
        if not cond:
            print(msg)
            self.gh.post(
                f"https://api.github.com/repos/{self.REPO}/issues/{self.NUMBER}/comments",
                json={
                    "body": f"ghstack bot failed: {msg}",
                },
            )
            exit(1)

    def shared_checks(self):
        self.head_ref = self.PR["head"]["ref"]
        self.must(
            self.head_ref
            and re.match(r"^gh/[A-Za-z0-9-]+/[0-9]+/head$", self.head_ref),
            "Not a ghstack PR",
        )
        self.orig_ref = self.head_ref.replace("/head", "/orig")
        print(":: Fetching newest main...")
        self.must(os.system("git fetch origin main") == 0, "Can't fetch main")
        print(":: Fetching orig branch...")
        self.must(
            os.system(f"git fetch origin {self.orig_ref}") == 0,
            "Can't fetch orig branch",
        )

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
            print(f":: Checking PR status #{n}... ", end="")
            resp = self.gh.get(f"https://api.github.com/repos/{self.REPO}/pulls/{n}")
            self.must(resp.ok, "Error Getting PR Object!")
            pr_obj = resp.json()

            resp = self.gh.get(
                f"https://api.github.com/repos/{self.REPO}/pulls/{self.NUMBER}/reviews"
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
                    f"@{u} has stamped PR #{n} `{cc}`, please resolve it first!",
                )

            self.must(approved, f"PR #{n} is not approved yet!")

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
                    f"PR #{n} check-run `{name}`'s status `{status}` is not success!",
                )
            print("SUCCESS!")

        print(":: All PRs are ready to be landed!")

    def rebase(self):
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

        print(":: All PRs are ready to be rebased!")


def main(check):
    # Setup the wrapper class
    EV = json.loads(sys.stdin.read())
    ghstack_checks = GHStackChecks(os.environ["GITHUB_TOKEN"], EV)

    # Run the github action specific checks
    if check == "land":
        ghstack_checks.land()
    elif check == "rebase":
        ghstack_checks.rebase()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHStack Permission Checks")
    parser.add_argument("check", choices=["land", "rebase"])
    args = parser.parse_args()

    main(args.check)
