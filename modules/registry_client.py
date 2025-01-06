from __future__ import annotations

import aiohttp
from apolo_sdk import RemoteImage
from yarl import URL


class RegistryV2Client:
    def __init__(
        self, token: str | None = None, base_url: URL = URL("https://ghcr.io/")
    ) -> None:
        self._base_url = base_url
        self._token = token

    async def list_repo_tags(self, owner: str, repo: str) -> list[RemoteImage]:
        headers = await self.repo_headers(owner, repo)
        url = self._base_url / "v2" / owner / repo / "tags" / "list"
        res = []
        async with aiohttp.ClientSession() as cl:
            async with cl.get(url, headers=headers) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()

        for tag in resp_json["tags"]:
            res.append(RemoteImage(f"{self._base_url.host}/{owner}/{repo}", tag=tag))
        return res

    async def repo_headers(self, owner: str, repo: str) -> dict[str, str]:
        token = self._token or await self.get_repo_anon_pull_token(owner, repo)
        return {"Authorization": f"Bearer {token}"}

    async def get_repo_anon_pull_token(self, owner: str, repo: str) -> str:
        async with aiohttp.ClientSession() as cl:
            # this is dirty, I didnt investigate further, but nvidia registry exposes
            # token endpoint at `proxy_auth`
            if self._base_url.host == "nvcr.io":
                url = self._base_url / "proxy_auth"
            else:
                url = self._base_url / "token"
            url = url.with_query({"scope": f"repository:{owner}/{repo}:pull"})
            async with cl.get(url) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()
                return resp_json["token"]
