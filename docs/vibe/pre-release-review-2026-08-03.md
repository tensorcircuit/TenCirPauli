# TenCirPauli Pre-release Review Report

审计日期：2026-08-03。审计范围：当前 `main` 工作区的功能验证、发行物构建、GitHub Actions、PyPI 发布链路、敏感信息与开源治理准备度。本轮未修改任何源码、测试、现有配置或现有文档；仅新增本审计报告。

## 总结结论

当前源码已经达到“Alpha 版本可以继续发布准备”的功能质量水平，但还没有达到“可以不经人工复核直接作为正式 GitHub/PyPI 发布链放行”的水平。最重要的两个问题是：本地没有 GitHub remote 或版本 tag，公开 PyPI 与 TestPyPI 查询不到 `tencirpauli`，因此无法确认用户所说的发布已经发生；从当前源码生成的 sdist 不包含 `LICENSE`，从该 sdist 再构建的 wheel 也不包含许可证文件。

## 已验证通过

- 工作区源码与测试：`scripts/check.py --benchmark smoke` 通过；Rust 27 tests、Python 187 passed/1 skipped、benchmark harness 133 passed/77 deselected；Rust fmt/Clippy、Black、Ruff 和 strict mypy 通过。
- Release extension：`maturin develop --release --locked` 通过；直接从当前源码构建的 macOS arm64 CPython 3.9 ABI3 wheel 包含当前全部 Python 模块，wheel import smoke 通过。
- Source build：从当前生成的 sdist 解包后，在干净的临时目录中用 `maturin build --release --locked` 成功构建。
- 功能示例：`examples/vqe_u1.py`、`examples/vqe_propagation.py`、`examples/vqe_spps.py` 和 `examples/tensorcircuit_interop.py` 均通过。
- Rust core 与 Python/TensorCircuit 边界符合当前架构目标；当前扫描未发现实际提交的 API key、密码、私钥、PyPI token 或 GitHub token。唯一的 `unsafe` 位于 U1 并行 pair kernel，当前有局部 safety 注释和实际并行阈值测试。
- `Cargo.lock` 已提交，`cargo tree --locked` 可复现；当前运行依赖检查 `pip check` 通过。
- 发布工作流采用 PyPI Trusted Publishing 的 OIDC 设计，不要求长期 `PYPI_API_TOKEN`；publish job 的权限范围是 `id-token: write`。

## 发布阻塞项

### P0-1：无法从当前仓库确认 GitHub/PyPI 已发布

证据：`git remote -v` 没有输出，当前没有 tag；`docs/vibe/releasing.md:3` 仍写着仓库没有 GitHub remote、没有进行 PyPI 发布；本次对 `https://pypi.org/pypi/tencirpauli/json` 的查询返回 404，TestPyPI 同名项目也返回 404。

处理：人工确认实际 GitHub owner/repository、PyPI 项目名和使用的索引。如果项目确实已经在别的账号、项目名或私有索引中发布，需要把正确信息补回发布记录；如果尚未发布，应先完成下面的 P0-2，再创建 remote、tag 和 Release。

### P0-2：sdist 缺少许可证文件

证据：当前运行 `maturin sdist` 生成的 `tencirpauli-0.1.0.tar.gz` 不含 `LICENSE`；从该 sdist 再构建的 wheel 也不含 `LICENSE`。直接从源码构建的 wheel 则包含 `tencirpauli-0.1.0.dist-info/licenses/LICENSE`。因此不能只检查直接 wheel，必须修复并检查 sdist 路径。

处理：调整 maturin/Cargo/Python 的许可证文件打包配置，使 sdist 和从 sdist 重建的 wheel 都包含许可证；重新构建并检查 archive contents、metadata 和 clean-install smoke。若同一版本已经上传过错误文件，PyPI 不允许覆盖，必须递增版本号。

### P0-3：不要使用当前本地 `dist/` 目录直接上传

`dist/` 被 `.gitignore` 忽略，当前目录中的 0.1.0 wheel/sdist 是旧构建物；其 metadata 仍是 Pre-Alpha，TensorCircuit 被标成 optional，且 wheel 只包含早期的少量 Python 模块，与当前 `pyproject.toml` 和源码不一致。发布前必须使用干净目录从目标 commit 重新构建，不能执行 `twine upload dist/*` 来上传这批旧文件。

## 重要但不立即阻塞的问题

- `.github/workflows/release.yml:20-24,39-43,53-57` 使用可变的 action major/tag 引用，没有 pin 到完整 commit SHA；对拥有 PyPI OIDC 发布权限的 workflow，建议固定 checkout、maturin、artifact 和 PyPI publish action 的 SHA，并定期更新。
- `.github/workflows/release.yml:78-93` 在 publish 前只等待构建 job，没有安装生成的 wheel、检查 sdist、检查许可证、运行 import/version smoke 或校验 release tag 与 `pyproject.toml` 版本相同；错误 tag 也可能构建出不匹配的包版本。应增加 pre-publish artifact QA 和 tag/version 一致性检查。
- `.github/workflows/ci.yml` 的 Rust 和 maturin 检查没有统一使用 `--locked`；release workflow 使用了 `--locked`，但日常 CI 因而不能完全验证锁文件约束。
- `pyproject.toml:5-21` 没有 `project.urls`，作者仍是泛化的 `TenCirPauli Contributors`；`Cargo.toml:8-13` 以及两个 crate manifest 也缺少 repository/homepage/documentation 元数据，maturin sdist 构建会提示该问题。发布前应补充真实仓库 URL、维护者信息、支持入口和 crate 元数据。
- `CHANGELOG.md:7-36` 把当前大量已实现能力放在 `Unreleased`，而 0.1.0 条目只描述较早的 Alpha facade；发布前应明确哪些内容属于 0.1.0，并记录最终 commit/tag。
- `docs/vibe/releasing.md:3` 与实际发布计划矛盾，`docs/vibe/implementation-status.md` 仍明确写着 Phase 6 under acceptance review 且有 concurrency、memory、matched-backend 和 benchmark handoff 剩余 gate。若发布 Alpha，应在 README 和 release notes 中明确这是 Alpha，不要暗示所有性能验收已完成。
- 跟踪文档中仍有 `/private/tmp` 等机器相关路径，例如 `benchmarks/README.md:5` 和 `docs/vibe/implementation-status.md:80,110`；不影响运行，但与项目“不把机器路径写入 tracked files”的规则不一致，应清理为抽象路径或普通示例。
- 当前没有 `SECURITY.md`、`CODEOWNERS`、Dependabot 配置、Issue templates、`CITATION.cff` 或 Code of Conduct。这些不是 Python 功能阻塞项，但对于公开仓库的维护和安全响应很有价值。
- 当前没有安装 `cargo-audit` 或 `pip-audit`，因此本次只完成了锁文件、依赖树和静态敏感信息检查，没有完成带 CVE 数据库的依赖漏洞审计；应在 CI 或定期维护任务中补上，并记录工具版本与结果。

## 功能与安全判断

功能方面，当前源码和本地 release 验证结果良好，公共 Pauli、Hamiltonian、grouping、symmetry、U1、deterministic propagation、SPPS、TensorCircuit conversion 和 examples 都有测试或 smoke evidence。它更准确的定位是“测试充分的 Alpha 版本”，而不是已经完成所有性能和长期兼容性承诺的稳定版；`Development Status :: 3 - Alpha` 与 `0.1.0` 是一致的。

安全方面，没有发现当前可见源码或可达 Git 历史中的真实密钥。工作流采用 OIDC 是正确方向，但真正的安全边界取决于 GitHub environment、release/tag 权限、branch protection、action SHA pinning 和 PyPI trusted publisher 的精确绑定；这些无法仅从本地文件确认。所有现有 Git 提交的作者邮箱是 `kcanamgal@foxmail.com`，推送到公开 GitHub 后会公开，需由维护者确认是否接受这一隐私结果。

## 建议执行顺序

1. 确认真实 GitHub/PyPI 项目身份，停止使用旧 `dist/` 文件。
2. 修复 sdist/LICENSE 问题，并做直接源码 wheel、sdist、从 sdist 重建 wheel、clean install、import/version、metadata 和许可证检查。
3. 修正版本、CHANGELOG、release 文档、仓库 URL、作者/维护者入口和机器路径。
4. 加强 release workflow：固定 action SHA、校验 tag/version、发布前安装 artifacts 并运行 smoke，明确 workflow 和 environment 权限。
5. 配置 GitHub branch/tag protection、`pypi` environment reviewer、PyPI Trusted Publisher、2FA、secret scanning、Dependabot 和安全响应入口。
6. 在干净环境中从实际 PyPI 安装并测试 Python 3.9、3.11、3.13；检查 Linux x86_64/aarch64、macOS x86_64/aarch64 和 Windows x64 wheel 后再正式发布。

## 我可以完成与必须手动完成的部分

我可以在你确认后直接修改仓库内的 sdist/LICENSE 打包配置、版本和 CHANGELOG、发布文档、项目 URL/维护者元数据、release workflow 的 artifact QA/tag 校验/action pinning、治理文件模板和本地路径问题，并重新执行完整验证。当前审计阶段没有做这些修改。

你必须手动完成 GitHub 建仓和 remote/push、tag 与 Release、branch/tag protection、Actions environment 和 required reviewers、PyPI 项目认领或 pending trusted publisher、Trusted Publisher 的 owner/repository/workflow/environment 精确配置、2FA 和账户恢复设置、secret scanning 等 GitHub 设置，以及版权/作者/联系方式/商标/引用信息的最终确认。

推荐使用 OIDC，不需要任何长期 PyPI token。如果不得不使用 token，应创建项目级最小权限 token，只放入 GitHub environment secret，绝不提交到仓库、写入 workflow 明文或通过聊天发送；发布后按轮换策略管理并在不需要时撤销。
