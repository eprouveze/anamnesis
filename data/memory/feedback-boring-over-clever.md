---
name: prefer the boring, well-exercised option
description: a fallback is only as real as its last verified exercise
type: feedback
---
Under failure, the boring option (a managed service with restart-on-crash) beat the
clever one (a bespoke sync). A fallback that has never actually run is indistinguishable
from a broken one until the day you need it. Exercise failover on a schedule.
