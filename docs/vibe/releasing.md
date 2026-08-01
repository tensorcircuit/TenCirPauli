# TenCirPauli 发布流程

状态：初始发布方案。当前仓库还没有 GitHub remote，也没有进行过 PyPI 发布。

## 两类工作流

`.github/workflows/ci.yml` 是日常 CI。每次 push 和 pull request 都会触发，但它只检查代码，不发布任何内容。Rust job 在 Ubuntu 上运行 rustfmt、Clippy 和 workspace tests；Python job 在 Linux、macOS、Windows 以及 Python 3.9、3.13 上用 `maturin develop --release` 编译并安装扩展，然后运行 pytest。这里的编译是安装/测试步骤，不会把 wheel 上传为 artifact，更不会上传 PyPI。

`.github/workflows/release.yml` 是发行工作流。手动触发 `workflow_dispatch` 时只构建并保存 wheel/sdist artifacts，用于发布前检查；只有 GitHub Release 的 `published` 事件才会执行 PyPI publish job。因此普通 commit、push 和 pull request 不可能触发 PyPI 发布。

## 一次性配置

推荐使用 PyPI Trusted Publishing，不保存长期 API token。先把仓库推送到 GitHub，然后在 GitHub 仓库 Settings → Environments 创建名为 `pypi` 的 environment；正式项目可以为它增加 required reviewers。接着在 PyPI 项目的 Publishing 设置中添加 GitHub publisher，填写 GitHub owner、repository、workflow 文件名 `release.yml` 和 environment `pypi`。第一次发布时，如果 PyPI 项目尚不存在，可以使用 PyPI 的 pending trusted publisher 流程预先登记这些信息。

GitHub Actions 的 publish job 只有 `id-token: write` 权限，`pypa/gh-action-pypi-publish` 会用短期 OIDC identity 换取上传权限，不需要配置 `PYPI_API_TOKEN`。仓库默认权限应保持 read-only；发布 job 单独申请 OIDC 权限。

## 正式发布步骤

1. 同步更新 workspace `Cargo.toml` 与 `pyproject.toml` 的版本号，并更新 `CHANGELOG.md`。
2. 完成日常 CI，创建并推送版本 tag，例如 `v0.1.0`。
3. 在 GitHub 中基于该 tag 创建 Release；先保存 draft 可检查内容，点击 Publish release 后才触发发行工作流。
4. 工作流分别构建 Linux x86_64/aarch64、macOS x86_64/aarch64、Windows x64 wheel 和一个 sdist，汇总成功后才发布到 PyPI。
5. 发布后从 PyPI 在干净环境中安装一次，执行 import/version smoke test。

PyPI 不允许覆盖同一版本文件，因此版本号和 tag 应视为不可变。若某次发布有问题，应修复后增加版本号，而不是重传相同版本。
