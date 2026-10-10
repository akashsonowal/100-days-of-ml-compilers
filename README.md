# 100 Days of ML Compilers

This repository is a practical, reusable roadmap for learning ML compilers from the ground up. It is designed for learners who want a structured path through the most important systems, concepts, and papers used in modern machine learning compilation.

Whether you are studying independently, preparing for interviews, or building a deeper systems understanding, this plan is meant to be clear, self-contained, and easy to revisit in short revision bursts.

The goal is to become strong enough to:
- understand compiler fundamentals for ML workloads,
- reason about IRs, lowering, scheduling, fusion, and memory optimization,
- compare major stacks such as TVM, XLA, PyTorch compiler, MLIR, Triton, IREE, and Halide,
- connect theory to real compiler systems and production ML workflows,
- use the revision logs as a lightweight knowledge base for future refreshes.

This repo is not just a reading list; it is a study system that helps you learn, retain, and revisit the material efficiently.

## Core resources

Primary references from this study track:
- MLC book: https://book.mlc.ai/
- TVM documentation: https://tvm.d2l.ai/index.html
- APXML courses:
  - https://apxml.com/courses/intro-ml-compiler-optimization
  - https://apxml.com/courses/compiler-optimizations-machine-learning
  - https://apxml.com/courses/compiler-runtime-optimization-ml
- PyTorch compiler docs: https://docs.pytorch.org/assets/pytorch2-2.pdf
- XLA: https://openxla.org/
- MLIR ecosystem and compilers: IREE, Triton, PolyBlocks, Nova
- Halide paper: https://people.csail.mit.edu/jrk/halide-pldi13.pdf
- TVM paper: https://arxiv.org/pdf/1802.04799
- nvFuser: https://pytorch.org/blog/introducing-nvfuser-a-deep-learning-compiler-for-pytorch/
- NNC: https://dev-discuss.pytorch.org/t/nnc-walkthrough-how-pytorch-ops-get-fused/125
- GRACE: https://www.csa.iisc.ac.in/~arkapravab/papers/OSDI26_GraCE.pdf
- StreamAlloc: https://dl.acm.org/doi/pdf/10.1145/3799893
- Triton compiler paper: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- Triton follow-up paper: https://arxiv.org/pdf/2504.19442
- IREE: https://iree.dev/
- Nova: https://arxiv.org/pdf/2608.00029
- PolyBlocks: https://arxiv.org/pdf/2603.06731

## Study philosophy

This is not a pure theory track. Every phase combines:
- concept study,
- reading a paper or design doc,
- implementing or tracing one small example,
- writing a short note on what changed in your mental model.

You should keep a daily log in the revision folder and end each week with a 15-minute recap. The goal is to make learning cumulative and review-friendly, not just linear and exhausting.

---

## 100-day roadmap

### Phase 1: Foundations of compilers and ML execution (Days 1-20)

Focus: build the mental model before looking at ML-specific systems.

Topics:
- compiler pipeline: frontend, IR, optimization, codegen
- SSA form and dataflow analysis
- loop transformations: fusion, tiling, unrolling, vectorization
- memory hierarchy and bandwidth bottlenecks
- graph representation of neural networks
- parallelism: data parallel, model parallel, pipeline parallel
- ML workloads: GEMM, convolution, attention, reduction, memory access patterns

Deliverables:
- one-page summary of compiler phases
- one-page summary of loop and memory optimization concepts
- annotated notes from the MLC book and APXML intro course

Key readings:
- MLC book chapters on compiler fundamentals
- APXML intro course
- Halide and TVM as examples of schedule-driven compilation

### Day-by-day breakdown for Phase 1

1. Day 1: Set up the repo and define your learning goals. Read the MLC overview.
2. Day 2: Learn the compiler pipeline: frontend, IR, optimization, codegen.
3. Day 3: Study SSA form and why it matters for optimization.
4. Day 4: Learn dataflow analysis and control-flow basics.
5. Day 5: Understand machine learning execution: eager mode, graph mode, compiled mode.
6. Day 6: Study loop transformations: fusion, tiling, unrolling, vectorization.
7. Day 7: Learn memory hierarchy and bandwidth bottlenecks.
8. Day 8: Understand GEMM, conv, reduction, and attention patterns.
9. Day 9: Read APXML intro material and summarize major themes.
10. Day 10: Write a one-page summary: “How compilers optimize ML workloads differently.”
11. Day 11: Learn graph optimization vs kernel optimization.
12. Day 12: Study parallelism: data, model, and pipeline parallelism.
13. Day 13: Understand latency and throughput in ML systems.
14. Day 14: Read the Halide paper introduction and extract the scheduling idea.
15. Day 15: Compare Halide with a standard compiler flow.
16. Day 16: Review compiler phases and refine your mental model.
17. Day 17: Re-read the MLC chapters on optimization and summarize them.
18. Day 18: Write notes on memory bottlenecks and compute bottlenecks.
19. Day 19: Build a first compiler cheat sheet with IR, optimization, and codegen.
20. Day 20: Mini assessment: explain the pipeline, memory bottlenecks, and ML workloads.

---

### Phase 2: Tensor compiler basics and scheduling (Days 21-40)

Focus: move from generic compiler concepts to machine learning tensor compilers.

Topics:
- kernel fusion and operation fusion
- scheduling concepts: tile, reorder, split, vectorize, unroll
- IR lowering from high-level tensor ops to loops
- autotuning and search space
- static vs dynamic shapes
- tensor expressions and schedule languages
- performance modeling and roofline thinking

Deliverables:
- explain the difference between graph optimization and kernel scheduling
- build a small schedule example using TVM or a toy tensor kernel
- compare compiler optimization from a loop nest and from an ML graph

Key readings:
- TVM paper
- TVM d2l tutorial
- APXML compiler optimization course

### Day-by-day breakdown for Phase 2

21. Day 21: Start TVM d2l and learn the high-level idea of tensor compilers.
22. Day 22: Learn schedule primitives: split, fuse, reorder, tile, vectorize.
23. Day 23: Study how operators map into loop nests.
24. Day 24: Understand autotuning and the optimization search space.
25. Day 25: Write a notebook summary of TVM schedule concepts.
26. Day 26: Read the TVM paper and identify the key innovation.
27. Day 27: Study graph-level optimization vs kernel-level scheduling.
28. Day 28: Learn static vs dynamic shapes and shape specialization.
29. Day 29: Review APXML optimization lectures and connect them to TVM.
30. Day 30: Write a comparison of generic compiler optimization vs tensor compiler scheduling.
31. Day 31: Understand tiling in detail and derive one example.
32. Day 32: Learn why data locality matters for transformer and CNN workloads.
33. Day 33: Study memory planning and runtime scheduling in tensor compilers.
34. Day 34: Read a small TVM example and write pseudo-code for its schedule.
35. Day 35: Summarize “what makes TVM different from a regular compiler.”
36. Day 36: Learn the role of tensor expression languages.
37. Day 37: Compare schedule-driven compilation with opaque backend compilers.
38. Day 38: Read one more TVM tutorial and capture 3 practical insights.
39. Day 39: Revisit days 21–38 and identify weak concepts to revisit.
40. Day 40: Midpoint assessment: explain TVM, schedule, fusion, and autotuning.

---

### Phase 3: PyTorch compiler stack (Days 41-60)

Focus: understand the end-to-end compiler landscape used by real ML frameworks.

Topics:
- torch.compile overview and execution model
- graph capture and backend dispatch
- fusion in PyTorch and graph-level optimizations
- nvFuser and kernel fusion
- NNC and operator fusion patterns
- GRACE and memory-centric compiler ideas
- StreamAlloc and runtime memory management

Deliverables:
- compare eager mode, compiled mode, and backend-specific execution
- trace the compiler stack from Python op to fused kernel
- write notes comparing nvFuser, NNC, and torch.compile behavior

Key readings:
- PyTorch 2.2 compiler whitepaper
- nvFuser blog post
- NNC walkthrough
- GRACE paper
- StreamAlloc paper

### Day-by-day breakdown for Phase 3

41. Day 41: Read the PyTorch 2.2 compiler paper and understand the big picture.
42. Day 42: Learn torch.compile flow: graph capture, partitioning, backend dispatch.
43. Day 43: Study operator fusion in PyTorch and its impact on memory traffic.
44. Day 44: Learn the difference between eager mode and compiled execution.
45. Day 45: Draw a diagram of the PyTorch compiler path from Python op to generated kernel.
46. Day 46: Read nvFuser and understand fusion in a backend compiler.
47. Day 47: Learn NNC and the fusion of PyTorch ops into kernels.
48. Day 48: Compare nvFuser vs NNC in one page.
49. Day 49: Read the Grace paper summary and extract the main insight.
50. Day 50: Write a note: “What is new in compiler-aware memory management?”
51. Day 51: Study StreamAlloc and runtime memory allocation.
52. Day 52: Learn how runtime allocation interacts with fusion and scheduling.
53. Day 53: Review the full PyTorch compiler stack and write a clean summary.
54. Day 54: Compare torch.compile, nvFuser, NNC, and runtime allocation in one table.
55. Day 55: Explain what makes PyTorch’s compiler stack practical for production ML.
56. Day 56: Study backend-specific optimization and dispatch.
57. Day 57: Learn the difference between compiler-generated kernels and handwritten kernels.
58. Day 58: Review all PyTorch notes and identify blurry concepts.
59. Day 59: Write a one-page summary of “PyTorch compiler design choices.”
60. Day 60: Quiz yourself on PyTorch compiler, fusion, and runtime management.

---

### Phase 4: XLA, MLIR, Triton, and the lowering ecosystem (Days 61-80)

Focus: understand how modern ML compilers lower and specialize workloads.

Topics:
- XLA architecture, HLO, runtime, and compilation flow
- MLIR fundamentals: dialects, passes, and lowering
- structured transformations and IR composition
- Triton language and GPU kernel compilation
- IREE and end-to-end MLIR-based execution
- PolyBlocks and advanced compiler generation ideas
- Nova and modern research directions

Deliverables:
- explain how XLA and MLIR differ from TVM in compilation philosophy
- summarize the role of dialects and passes in MLIR
- write a comparison between Triton, IREE, and TVM for GPU kernels

Key readings:
- XLA docs and architecture overview
- MLIR concepts and dialect tutorial
- Triton compiler papers
- IREE documentation
- PolyBlocks paper
- Nova paper

### Day-by-day breakdown for Phase 4

61. Day 61: Read the XLA overview and learn the compiler architecture and HLO.
62. Day 62: Understand XLA’s approach to graph optimization and lowering.
63. Day 63: Compare XLA and TVM at the system level.
64. Day 64: Study the role of graph lowering in ML frameworks.
65. Day 65: Write a comparison: “XLA vs TVM vs torch.compile.”
66. Day 66: Learn MLIR basics: dialects, passes, lowering, and IR composition.
67. Day 67: Understand why MLIR is useful for ML compiler stacks.
68. Day 68: Study dialects for tensor operations and memory semantics.
69. Day 69: Learn structured transformations and pass pipelines.
70. Day 70: Write one page explaining why MLIR matters for compiler reuse.
71. Day 71: Read the Triton compiler paper and extract the main idea.
72. Day 72: Learn Triton kernels and GPU specialization.
73. Day 73: Study the relation between Triton and MLIR infrastructure.
74. Day 74: Read IREE documentation and summarize the stack.
75. Day 75: Compare Triton vs IREE vs TVM for GPU kernels.
76. Day 76: Read the Nova paper and understand the modern research direction.
77. Day 77: Read PolyBlocks paper summary and extract the core idea.
78. Day 78: Compare research compilers: Nova, PolyBlocks, and Triton.
79. Day 79: Build a final “compiler ecosystem map” covering XLA, TVM, torch.compile, MLIR, Triton, and IREE.
80. Day 80: Write a summary of your understanding of modern ML compiler infrastructure.

---

### Phase 5: Systems, research, and project consolidation (Days 81-100)

Focus: connect all ideas into a real working understanding.

Topics:
- compiler scheduling for attention and transformer workloads
- memory planner and allocator design
- advanced fusion and graph partitioning
- compiler runtime integration
- benchmarking and performance debugging
- reading compiler papers and extracting core ideas

Deliverables:
- end-to-end notes comparing PyTorch, TVM, XLA, and MLIR paths
- one mini project: a small optimization writeup, a reproduction note, or a toy compiler pass
- final 2-day revision pack for future refreshes

Suggested final projects:
- write a short report comparing TVM vs XLA vs torch.compile
- implement a toy tensor kernel with scheduling and explain each pass
- build a compiler note pack summarizing the main papers and systems

### Day-by-day breakdown for Phase 5

81. Day 81: Review the full compiler pipeline and write the cleanest version you can.
82. Day 82: Review fusion, tiling, and scheduling and write three examples each.
83. Day 83: Study memory planning and runtime allocation again.
84. Day 84: Note how attention and transformer workloads stress compilers differently.
85. Day 85: Re-read a key paper and summarize it in one page.
86. Day 86: Compare Halide, TVM, and Triton as different compiler philosophies.
87. Day 87: Compare XLA, PyTorch compiler, and MLIR-driven stacks.
88. Day 88: Write a short report on “what makes an ML compiler efficient.”
89. Day 89: Research checkpoint: list the top 10 ideas you understand deeply.
90. Day 90: Review all notes and identify the biggest remaining gaps.
91. Day 91: Build a final mini-project: short report, code example, or compiler diagram.
92. Day 92: Write a final comparison of graph optimization and kernel scheduling.
93. Day 93: Revisit the main papers and rank them by importance.
94. Day 94: Practice explaining the whole field in under 10 minutes.
95. Day 95: Write the final one-page “ML compilers mental model.”
96. Day 96: Begin the first revision pack: fundamentals and core concepts.
97. Day 97: Continue the revision pack: systems and compiler flows.
98. Day 98: Final paper recap and system comparison.
99. Day 99: Final self-test: answer 10 important ML compiler questions from memory.
100. Day 100: End-of-track review and decide what to revisit next.

---

## Weekly rhythm

Each week should include:
- 4 focused study blocks of 60-90 minutes
- 1 reading session from a paper or documentation
- 1 summary note of 5-10 bullet points
- 1 comparison table between two systems

Recommended weekly outputs:
- one concept summary
- one system comparison
- one code or kernel trace
- one review note for future revision

---

## Daily study template

Use this pattern every study day:

1. Read or watch 20-40 minutes of material.
2. Write down 3 takeaways.
3. Create 2 questions that remain unanswered.
4. Note one concrete optimization concept or compiler pass.
5. Add a short code or pseudocode example.

---

## 1-2 day revision mode

When you only have 1-2 days before a review or interview, use this compressed plan:

### Day 1: Core concepts
- compiler pipeline and stages
- IR and SSA
- fusion, loop transformations, tiling
- memory hierarchy and performance bottlenecks
- graph vs kernel optimization

### Day 2: System comparison and research focus
- TVM vs XLA vs torch.compile
- MLIR and Triton lowering story
- key papers: TVM, Triton, nvFuser, GRACE, Halide
- runtime memory and scheduling hacks
- final recap and 10-question self-test

This repo is meant to be revisited in that short format. Keep your notes concise and use the revision log to refresh without restarting from zero.

---

## Quick reference: major systems

- TVM: schedule-driven tensor compiler, strong for autotuning and kernel generation.
- XLA: compiler for TensorFlow and JAX ecosystems, focuses on graph lowering and HLO.
- PyTorch compiler: eager + graph capture + backend-specific compilation with torch.compile.
- MLIR: reusable compiler infrastructure with dialects and passes.
- Triton: GPU-focused programming language and compiler for high-performance kernels.
- IREE: MLIR-based runtime and compiler stack for heterogeneous execution.
- Halide: domain-specific compiler for image and stencil workloads with schedule separation.

---

## Suggested milestone checkpoints

- Day 20: understand compiler pipeline and optimization basics
- Day 40: able to explain scheduling and tensor compilation at a high level
- Day 60: can compare PyTorch compiler subsystems and fusion strategies
- Day 80: can explain MLIR/XLA/Triton lowering flows
- Day 100: can articulate research-level distinctions between leading ML compiler stacks

---

## Revision log usage

Use the files in the logs directory as your working notebook:
- add a daily entry when training,
- maintain a 1-2 day revision entry before interviews or code reviews,
- keep a short list of key concepts and paper summaries,
- review the same notes repeatedly until they become high-confidence recall.

---

## Resource list summary

- MLC book
- TVM d2l
- APXML compiler optimization series
- PyTorch 2.2 compiler notes
- XLA docs
- MLIR docs
- Triton papers
- IREE docs
- TVM paper
- Halide paper
- nvFuser, NNC, GRACE, StreamAlloc, PolyBlocks, Nova

This repo is intentionally structured for revision: quick to scan, focused on the highest-yield ideas, and easy to revisit in a compressed study window. It is designed to be useful for many learners, whether they are beginning the journey or doing a fast refresher before interviews and systems work.

---

## How to use this repo effectively

1. Follow the day-by-day plan consistently.
2. Update your daily study log after each session.
3. End each week with a recap and a system comparison.
4. Use the revision templates when you need a faster refresh in 1–2 days.
5. Return to the repo periodically as a reference while working on ML systems and compiler-related projects.

This repository is intended to serve as a reusable learning and revision system for the ML compilers community and anyone preparing to work at the intersection of compilers, kernels, and modern ML systems.

### Phase 1: Foundations of compilers and ML execution (Days 1-20)

Focus: build the mental model before looking at ML-specific systems.

Topics:
- compiler pipeline: frontend, IR, optimization, codegen
- SSA form and dataflow analysis
- loop transformations: fusion, tiling, unrolling, vectorization
- memory hierarchy and bandwidth bottlenecks
- graph representation of neural networks
- parallelism: data parallel, model parallel, pipeline parallel
- ML workloads: GEMM, convolution, attention, reduction, memory access patterns

Deliverables:
- one-page summary of compiler phases
- one-page summary of loop and memory optimization concepts
- annotated notes from the MLC book and APXML intro course

Key readings:
- MLC book chapters on compiler fundamentals
- APXML intro course
- Halide and TVM as examples of schedule-driven compilation

---

### Phase 2: Tensor compiler basics and scheduling (Days 21-40)

Focus: move from generic compiler concepts to machine learning tensor compilers.

Topics:
- kernel fusion and operation fusion
- scheduling concepts: tile, reorder, split, vectorize, unroll
- IR lowering from high-level tensor ops to loops
- autotuning and search space
- static vs dynamic shapes
- tensor expressions and schedule languages
- performance modeling and roofline thinking

Deliverables:
- explain the difference between graph optimization and kernel scheduling
- build a small schedule example using TVM or a toy tensor kernel
- compare compiler optimization from a loop nest and from an ML graph

Key readings:
- TVM paper
- TVM d2l tutorial
- APXML compiler optimization course

---

### Phase 3: PyTorch compiler stack (Days 41-60)

Focus: understand the end-to-end compiler landscape used by real ML frameworks.

Topics:
- torch.compile overview and execution model
- graph capture and backend dispatch
- fusion in PyTorch and graph-level optimizations
- nvFuser and kernel fusion
- NNC and operator fusion patterns
- GRACE and memory-centric compiler ideas
- StreamAlloc and runtime memory management

Deliverables:
- compare eager mode, compiled mode, and backend-specific execution
- trace the compiler stack from Python op to fused kernel
- write notes comparing nvFuser, NNC, and torch.compile behavior

Key readings:
- PyTorch 2.2 compiler whitepaper
- nvFuser blog post
- NNC walkthrough
- GRACE paper
- StreamAlloc paper

---

### Phase 4: XLA, MLIR, Triton, and the lowering ecosystem (Days 61-80)

Focus: understand how modern ML compilers lower and specialize workloads.

Topics:
- XLA architecture, HLO, runtime, and compilation flow
- MLIR fundamentals: dialects, passes, and lowering
- structured transformations and IR composition
- Triton language and GPU kernel compilation
- IREE and end-to-end MLIR-based execution
- PolyBlocks and advanced compiler generation ideas
- Nova and modern research directions

Deliverables:
- explain how XLA and MLIR differ from TVM in compilation philosophy
- summarize the role of dialects and passes in MLIR
- write a comparison between Triton, IREE, and TVM for GPU kernels

Key readings:
- XLA docs and architecture overview
- MLIR concepts and dialect tutorial
- Triton compiler papers
- IREE documentation
- PolyBlocks paper
- Nova paper

---

### Phase 5: Systems, research, and project consolidation (Days 81-100)

Focus: connect all ideas into a real working understanding.

Topics:
- compiler scheduling for attention and transformer workloads
- memory planner and allocator design
- advanced fusion and graph partitioning
- compiler runtime integration
- benchmarking and performance debugging
- reading compiler papers and extracting core ideas

Deliverables:
- end-to-end notes comparing PyTorch, TVM, XLA, and MLIR paths
- one mini project: a small optimization writeup, a reproduction note, or a toy compiler pass
- final 2-day revision pack for future refreshes

Suggested final projects:
- write a short report comparing TVM vs XLA vs torch.compile
- implement a toy tensor kernel with scheduling and explain each pass
- build a compiler note pack summarizing the main papers and systems

---

## Weekly rhythm

Each week should include:
- 4 focused study blocks of 60-90 minutes
- 1 reading session from a paper or documentation
- 1 summary note of 5-10 bullet points
- 1 comparison table between two systems

Recommended weekly outputs:
- one concept summary
- one system comparison
- one code or kernel trace
- one review note for future revision

---

## Daily study template

Use this pattern every study day:

1. Read or watch 20-40 minutes of material.
2. Write down 3 takeaways.
3. Create 2 questions that remain unanswered.
4. Note one concrete optimization concept or compiler pass.
5. Add a short code or pseudocode example.

---

## 1-2 day revision mode

When you only have 1-2 days before a review or interview, use this compressed plan:

### Day 1: Core concepts
- compiler pipeline and stages
- IR and SSA
- fusion, loop transformations, tiling
- memory hierarchy and performance bottlenecks
- graph vs kernel optimization

### Day 2: System comparison and research focus
- TVM vs XLA vs torch.compile
- MLIR and Triton lowering story
- key papers: TVM, Triton, nvFuser, GRACE, Halide
- runtime memory and scheduling hacks
- final recap and 10-question self-test

This repo is meant to be revisited in that short format. Keep your notes concise and use the revision log to refresh without restarting from zero.

---

## Quick reference: major systems

- TVM: schedule-driven tensor compiler, strong for autotuning and kernel generation.
- XLA: compiler for TensorFlow and JAX ecosystems, focuses on graph lowering and HLO.
- PyTorch compiler: eager + graph capture + backend-specific compilation with torch.compile.
- MLIR: reusable compiler infrastructure with dialects and passes.
- Triton: GPU-focused programming language and compiler for high-performance kernels.
- IREE: MLIR-based runtime and compiler stack for heterogeneous execution.
- Halide: domain-specific compiler for image and stencil workloads with schedule separation.

---

## Suggested milestone checkpoints

- Day 20: understand compiler pipeline and optimization basics
- Day 40: able to explain scheduling and tensor compilation at a high level
- Day 60: can compare PyTorch compiler subsystems and fusion strategies
- Day 80: can explain MLIR/XLA/Triton lowering flows
- Day 100: can articulate research-level distinctions between leading ML compiler stacks

---

## Revision log usage

Use the files in the logs directory as your working notebook:
- add a daily entry when training,
- maintain a 1-2 day revision entry before interviews or code reviews,
- keep a short list of key concepts and paper summaries,
- review the same notes repeatedly until they become high-confidence recall.

---

## Resource list summary

- MLC book
- TVM d2l
- APXML compiler optimization series
- PyTorch 2.2 compiler notes
- XLA docs
- MLIR docs
- Triton papers
- IREE docs
- TVM paper
- Halide paper
- nvFuser, NNC, GRACE, StreamAlloc, PolyBlocks, Nova

This repo is intentionally structured for revision: quick to scan, focused on the highest-yield ideas, and easy to revisit in a compressed study window.


