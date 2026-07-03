# Contributing to ProcessTensors.jl

First of all...

**Thank you for even considering contributing.** ❤️

Whether you found a typo, fixed a bug, improved an example, discovered a physics mistake, or implemented an entirely new algorithm—you are helping make this project better for everyone.

This package exists because open quantum systems deserve better tensor-network tools in Julia, and building that ecosystem is very much a community effort.

---

## Before writing code...

Take a look around.

Read the documentation.

Browse the existing source code.

You may discover that the feature you were planning already exists... just under a name you didn't expect.

(We've all been there.)

---

## Found a bug?

Fantastic.

Well... not fantastic.

Please open an issue describing:

* what you expected,
* what actually happened,
* how to reproduce it,
* your Julia version,
* your package versions,
* and, if possible, a minimal working example.

The smaller the reproducer, the happier future-you (and future-us) will be.

---

## Want to add a new feature?

Even better.

If it's a substantial addition, open an issue first.

A five-minute discussion can save five days of implementing something that ends up not fitting the package direction.

We're always excited about:

* better process-tensor algorithms,
* Liouville-space utilities,
* new instrument types,
* bath and environment models,
* tutorials,
* examples,
* documentation improvements,
* tests,
* benchmarks,
* performance improvements.

---

## Coding style

This project tries to write code that reads like physics.

A few guiding principles:

* Prefer multiple dispatch over large conditional blocks.
* Avoid unnecessary helper functions.
* Keep related logic together.
* Optimize for readability first, cleverness second.
* Write code that your future self will understand six months later.
* If a function needs a flowchart to understand... it probably needs simplification.

---

## Documentation

Good documentation is considered a feature.

If you add public functionality, please also update:

* the API documentation,
* relevant tutorials,
* examples when appropriate.

Remember:

> Someone reading your documentation today might become tomorrow's contributor.

---

## Tests

If you fix a bug...

...please add a test.

If you add a feature...

...please add a test.

If you optimize something...

...please make sure the existing tests still pass.

Physics is already difficult enough—we'd rather not debug quantum mechanics *and* regressions at the same time.

---

## Commit messages

Clear commit messages are greatly appreciated.

Good:

```text
Add Liouville MPO constructor for tuple jump operators
```

Also good:

```text
Improve TDVP example documentation
```

Less helpful:

```text
stuff
```

Or everyone's favorite:

```text
final_final_v7_REAL_THIS_ONE
```

---

## Pull Requests

Before opening a PR, please check:

* [ ] The code builds.
* [ ] Tests pass.
* [ ] Documentation is updated.
* [ ] Examples still work.
* [ ] Public APIs are documented.
* [ ] New code follows the existing style.

Small PRs are easier to review than giant "rewrite half the package" PRs.

---

## A note on discussions

Questions are welcome.

Ideas are welcome.

Disagreements are welcome.

Good scientific software improves through discussion, not ego.

If something can be explained with a benchmark, an equation, or a reference paper, that's even better.

---

## Research contributions

If you're implementing an algorithm from the literature, please include a reference whenever possible.

Future readers (and reviewers) will thank you.

---

## Finally...

Building scientific software is a strange hobby.

We spend weeks writing code so someone else can solve a physics problem in five minutes.

And honestly...

that's pretty awesome.

Thank you for helping make ProcessTensors.jl better.

Happy coding!
