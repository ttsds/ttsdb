from pathlib import Path
import docker

import yaml


class TTSProvider:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            self.config = yaml.safe_load(f)


providers = []

for yaml_file in (Path(__file__).parent / "providers").glob("*.yml"):
    providers.append(TTSProvider(yaml_file))

docker_client = docker.from_env()


class TTS:
    def __init__(self, system, version, provider=None, local=False, port=8888):
        self.system = system
        self.version = version
        if provider:
            self.provider = provider
        else:
            for p in providers:
                for s in p.config["systems"]:
                    print(s["id"], self.system)
                    if s["id"] == self.system:
                        if hasattr(self, "provider"):
                            raise ValueError(
                                f"Multiple providers found for {self.system}: {self.provider} and {p}"
                            )
                        self.provider = p
        if not hasattr(self, "provider"):
            raise ValueError(f"No provider found for {self.system}")
        self.local = local
        if self.local:
            docker_client.containers.run(
                self.provider.config["provider_name"],
                ports={f"{port}/tcp": port},
            )

    def download(self):
        print(f"Downloading {self.system} {self.version} from {self.provider}")


valle = TTS("valle", "v1_small", local=True)
