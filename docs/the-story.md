# The story: how this design was earned

This kit didn't come from a whiteboard. It came from an outage. The narrative matters more
than the code, so here it is — genericized, but real.

## The outage

An autonomous agent had been running for months against a persistent memory system. That
memory lived on a single **primary machine**: it was the sole writer, the sole indexer, and
the sole source that pushed copies to a few read replicas. One morning the primary dropped
off the network — a hard, host-level death, minutes after a healthy checkpoint.

Reads survived (the replicas kept serving a frozen copy). But **every write and every
refresh stopped**, because all of that machinery lived on the one machine that was now gone.
Worse: the failover layer that was supposed to cover this had been **silently broken for
weeks** — a keep-alive that pointed at a path that no longer existed, a replica that had
been stale for days with nobody watching. *A fallback that has never actually run is
indistinguishable from a working one until the day you need it.*

## The review

Rather than just patch it, the design was put through a real review:

- **A multi-model design council.** Several frontier models were asked the same sharp
  question — rebuild the memory so *no machine is required for the write path*, evaluated on
  maintenance burden for a solo operator, failure modes, and reversibility. They converged,
  nearly unanimously, on the same answer: **make the database a build artifact and let git
  be the write-ahead log.** (This is the [rightmodel](https://github.com/eprouveze/rightmodel)
  pattern — diverse independent judgment before a big commitment.)
- **A security pass.** Trading "one fragile primary" for "any machine can write to a
  shared, trusted memory" *enlarges* the attack surface: a single poisoned entry could
  become trusted context for every future agent session. That produced hard requirements —
  protect the pipeline that holds the embedding key, verify artifacts before serving them,
  quarantine the highest-trust writes, keep write identity attributable, and split
  read-only from write credentials. Those requirements are baked into this design (see
  [architecture.md](architecture.md#security-model)).

## The rebuild (this kit)

The insight that made it cheap: **the memory sources were already in git.** So the index is
*derived*, not primary. Writes become commits; a CI pipeline rebuilds the index; every
machine — including whichever one used to be "primary" — becomes a disposable replica. Lose
any machine, lose all of them, or lose the CI: a write is still just a commit, and the index
still rebuilds anywhere from the sources. The primary-machine dependency doesn't get
mitigated — it stops existing.

## The lessons (portable beyond memory)

1. **If your sources are in git, your database is a build artifact.** Treat it as one.
2. **A fallback is only as real as its last verified exercise.** Drill it, or assume it's dead.
3. **"Skip if unreachable" needs a freshness alarm on the consuming side.** Silent staleness is the failure that hides longest.
4. **One writer per dataset**, enforced structurally (git commit ordering) rather than by a promotion procedure you can get wrong.
5. **Never mix timestamp conventions in one column.** UTC everywhere, or freshness math lies.
6. **Enlarging the write surface enlarges the poisoning surface.** Design the trust controls in from the start, not after.
