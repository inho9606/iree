---
layout: default
permalink: Dialects/FlowDialect
parent: Dialect Definitions
title: "'flow' Dialect"
---

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'flow' Dialect
{: .no_toc }


A dialect designed to model execution data flow and partitioning.


The flow dialect is used to model regions of dense computation and the data
flow between them. MLIR value-semantic tensors are used as the primary data
type to allow SSA use-def to provide a bulk of the infrastructure required
to perform the computation partitioning and outlining.

The dialect is designed to ingest relatively high-level linear algebra via
XLA HLO ops (that also operate on the value-semantic tensor types) and
optionally MLIR standard ops for control flow and other actions. After
conversion of any higher-level ops that have special semantics in the flow
dialect, such as global variables, the rest are partitioned into regions
containing simple and compatible computations. Finally, outlining moves the
computations into executables and leaves only the execution flow encoded via
dispatch operations.

The primary unit of interest is a "dispatch region" containing compatible
computations that can be scheduled together efficiently (and safely).
"Compatible" here is specified as similarly shaped workloads that indicate
how many invocations a computation can be parallelized across when running
in a SPMD execution model. Though it depends on the particular runtime
backends this more concretely means things like the untiled workload
(or tiled workgroups) used in GPU dispatches or similar thread pool
executors.

After identification of the dispatchable regions a set of transformations
performs folding and simplification to reduce the total number of
dispatches. Heuristics are used in certain cases to more efficiently
schedule special ops (such as GEMM) and the design is amenable to profile-
guided analysis that can be added in the future.

The resulting outlined executable modules containing the dispatchable code
can be translated to one or more backends (such as SPIR-V for Vulkan, or
LLVM IR for running on the CPU, etc). The IR that is outlined is untouched
and in the input format (such as XLA HLO ops) allowing conversion using any
MLIR target that supports ingesting such input. A few special ops are used
to communicate statically available information such as the expected
workload size, shapes of inputs and outputs, etc.

1. TOC
{:toc}

## Operation definition

### `flow.dispatch.entry` (IREE::Flow::DispatchEntryOp)

defines an executable entry point for dispatch operations

Specifies an exported function with an externally-visible alias. Multiple
exports can reference the same internal function.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | ::mlir::StringAttr | string attribute
`function_ref` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute
`workload` | ::mlir::IntegerAttr | size_t

### `flow.dispatch` (IREE::Flow::DispatchOp)

a dispatch to an outlined dispatch region

Dispatches a workload to the specified executable function.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`executable` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute
`entry_point` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`workload` | index
`operands` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`results` | any type

### `flow.dispatch.region` (IREE::Flow::DispatchRegionOp)

partitioned region representing a dispatched workload

A closure that represents a functional dispatch unit. These perform
computations in a way that can be lowered to target executable formats such
as SPIR-V for execution.

Ops that are identified as "dispatchable" are grouped into dispatch regions
and compatible dispatch regions are folded together. What remains outside of
the dispatch regions is the glue required to schedule the work (commonly
referred to as "host" code, even if it doesn't run on an AP).

Dispatch regions are modeled using value semantics: it is assumed that all
arguments are read-only and that the dispatch regions themselves have no
side-effects.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`workload` | index
`args` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`results` | any type

### `flow.ex.stream.fragment` (IREE::Flow::ExStreamFragmentOp)

experimental op for defining formed stream regions

Represents a region where all of the dispatches are meant to target the
same execution stream. This will be replaced with a segmented verison.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`args` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`results` | any type

### `flow.executable_end` (IREE::Flow::ExecutableEndOp)

terminator pseudo-op for the executable op

Syntax:

```
operation ::= `flow.executable_end` attr-dict
```



### `flow.executable` (IREE::Flow::ExecutableOp)

generic executable module

An executable module containing one or more public functions. The contents
of the functions are safe to dispatch and can be lowered further to
target-specific backend IR representations.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | ::mlir::StringAttr | string attribute

### `flow.return` (IREE::Flow::ReturnOp)

return from a flow.dispatch_region

Syntax:

```
operation ::= `flow.return` attr-dict ($operands^ `:` type($operands))?
```


Returns the given values from the region and back to the host code.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operands` | any type

### `flow.tensor.clone` (IREE::Flow::TensorCloneOp)

performs a full tensor clone operation

Syntax:

```
operation ::= `flow.tensor.clone` $operand `:` type($result) attr-dict
```


Clones the input tensor into an identical output tensor.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`operand` | ranked tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.tensor.load` (IREE::Flow::TensorLoadOp)

loads a value from a tensor element

Syntax:

```
operation ::= `flow.tensor.load` $source (`[` $indices^ `]`)? `:` type($source) attr-dict-with-keyword
```


Returns the element at the given location from within the tensor.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`source` | ranked tensor of any type values
`indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | index or signless integer or floating-point or vector of any type values

### `flow.tensor.reshape` (IREE::Flow::TensorReshapeOp)

reshapes a tensor

Syntax:

```
operation ::= `flow.tensor.reshape` $source `:` type($source) `->` type($result) attr-dict
```


Reshapes a tensor to a new shape without modifying the contents.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`source` | ranked tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.tensor.slice` (IREE::Flow::TensorSliceOp)

slices out a subregion of a tensor

Syntax:

```
operation ::= `flow.tensor.slice` $source `[` $start_indices `for` $lengths `]` `:` type($source) `->`
              type($result) attr-dict
```


Clones a subregion of a tensor.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`source` | ranked tensor of any type values
`start_indices` | index
`lengths` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.tensor.splat` (IREE::Flow::TensorSplatOp)

splats a value into a shaped tensor

Syntax:

```
operation ::= `flow.tensor.splat` $value `:` type($result) attr-dict-with-keyword
```


Returns a tensor initialized to the given primitive value.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | index or signless integer or floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.tensor.store` (IREE::Flow::TensorStoreOp)

stores a value into a tensor element

Syntax:

```
operation ::= `flow.tensor.store` $value `,` $target (`[` $indices^ `]`)? `:` type($target)
              attr-dict-with-keyword
```


Returns a tensor with the element at the given index set to the given value.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | index or signless integer or floating-point or vector of any type values
`target` | ranked tensor of any type values
`indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.tensor.update` (IREE::Flow::TensorUpdateOp)

updates a tensor with the contents of another tensor

Syntax:

```
operation ::= `flow.tensor.update` $update `,` $target `[` $start_indices `]` `:` type($update) `->`
              type($result) attr-dict
```


Updates the target tensor with the contents of the update tensor at the
given offset indices.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`update` | ranked tensor of any type values
`target` | ranked tensor of any type values
`start_indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.variable.address` (IREE::Flow::VariableAddressOp)

returns an address reference to a variable

Syntax:

```
operation ::= `flow.variable.address` $variable attr-dict `:` type($result)
```


Returns the address of a variable as a typed reference. Can be used with the
variable load and store indirect ops.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`variable` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values or index or signless integer or floating-point

### `flow.variable.load.indirect` (IREE::Flow::VariableLoadIndirectOp)

loads a value from a global variable

Syntax:

```
operation ::= `flow.variable.load.indirect` $variable attr-dict `:` type($variable) `->` type($result)
```


Returns a copy of the variable value.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`variable` | ranked tensor of any type values or index or signless integer or floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.variable.load` (IREE::Flow::VariableLoadOp)

loads a value from a global variable

Syntax:

```
operation ::= `flow.variable.load` $variable attr-dict `:` type($result)
```


Returns a copy of the variable value.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`variable` | FlatSymbolRefAttr | symbol reference attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | ranked tensor of any type values

### `flow.variable` (IREE::Flow::VariableOp)

stateful variable declaration

Declares a persistent variable that maintains its value.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`sym_name` | ::mlir::StringAttr | string attribute
`type` | ::mlir::TypeAttr | any type attribute
`is_mutable` | ::mlir::UnitAttr | unit attribute
`initializer` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute
`initial_value` | ::mlir::Attribute | any attribute

### `flow.variable.store.indirect` (IREE::Flow::VariableStoreIndirectOp)

stores a value into a global variable

Syntax:

```
operation ::= `flow.variable.store.indirect` $value `,` $variable attr-dict `:` type($value) `->` type($variable)
```


Stores a copy of the value into a variable.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | ranked tensor of any type values
`variable` | ranked tensor of any type values or index or signless integer or floating-point

### `flow.variable.store` (IREE::Flow::VariableStoreOp)

stores a value into a global variable

Syntax:

```
operation ::= `flow.variable.store` $value `,` $variable attr-dict `:` type($value)
```


Stores a copy of the value into a variable.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`variable` | FlatSymbolRefAttr | symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | ranked tensor of any type values