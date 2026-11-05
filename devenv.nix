{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [
    pkgs.git
    pkgs.cocogitto
  ];

  languages.python = {
    enable = true;
    version = "3.14";
    venv.enable = true;
    uv.enable = true;
    uv.sync.enable = true;
    uv.sync.allGroups = true;
  };

  tasks = {
    "check:conventional-commit" = {
      exec = ''
        if [ -n "$CI" ]; then
          # PR checkouts are merge commits, which are not conventional commits.
          ${pkgs.cocogitto}/bin/cog check --ignore-merge-commits "origin/''${GITHUB_BASE_REF:-main}.."
        else
          ${pkgs.cocogitto}/bin/cog check main..
        fi
      '';
      before = ["devenv:enterTest"];
    };
    "check:python-lint" = {
      exec = ''
        uv run ruff format --check
        uv run ruff check
      '';
      before = ["check:code-lint"];
    };
    "check:types" = {
      exec = ''
        uv run ty check src/
      '';
      before = ["check:code-lint"];
    };
    "check:code-lint" = {
      before = ["devenv:enterTest"];
      after = [
        "check:python-lint"
        "check:types"
      ];
    };
    "check:pytest" = {
      exec = ''
        uv run pytest
      '';
    };
    "check:unit-tests" = {
      before = ["devenv:enterTest"];
      after = [
        "check:pytest"
      ];
    };
  };

  # See full reference at https://devenv.sh/reference/options/
}
