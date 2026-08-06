# Advanced API reference

This page documents the stability-sensitive implementation types that may appear as return values or be needed for specialized workflows. Ordinary user code should prefer the factories, operator `compile()` methods, and circuit facades documented on the [public API reference](api.md). Advanced names are public but their signatures and internal-buffer contracts may evolve under API review.

The `tencirpauli._native` module is private and is not an alternative import path. Advanced plans and engines are factory-created unless their docstring explicitly documents a safe constructor. Arrays owned by plans or engines are generally immutable; arrays returned by `apply()` and execution terminals follow the ownership and mutability contract stated in their method documentation.

The ordinary circuit facades no longer expose public `compile()` methods or circuit-plan classes. `GateTape`, `PropagationEngine`, and `SPPSEngine` remain advanced numerical APIs; their explicit parameter slots are low-level runtime buffers and are not the removed symbolic `Parameter`/`ParameterExpr` interface.

::: tencirpauli.advanced
    options:
      docstring_style: google
      docstring_section_style: table
      filters:
        - "!^_"
      inherited_members: false
      members: true
      members_order: source
      show_root_heading: true
      show_root_full_path: false
      show_signature_annotations: true
      show_source: false
