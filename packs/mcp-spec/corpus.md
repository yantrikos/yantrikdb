# mcp-spec corpus — MCP specification revision 2026-07-28

Authored from the specification pages at modelcontextprotocol.io on
2026-07-30, two days after the revision shipped. The delta records come
first because they are the pack's reason to exist: every current model's
training data describes the protocol this revision replaced.

## MCP 2026-07-28 removed the initialize handshake

The `initialize` request and `notifications/initialized` notification no
longer exist. MCP is now stateless: every request carries its protocol
version and client capabilities in `_meta`, and a server processes each
request independently with no memory of prior requests on the same
connection. A model that writes an initialize handshake is writing
protocol revision 2025-11-25 or earlier.

```json
"_meta": {
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {}
}
```

## MCP 2026-07-28 removed protocol sessions and Mcp-Session-Id

The `Mcp-Session-Id` header is gone from the Streamable HTTP transport,
and list endpoints (`tools/list`, `resources/list`, `prompts/list`) no
longer vary per connection. A server that needs cross-call state mints
explicit handles and passes them as ordinary tool arguments. A modern
server receiving `Mcp-Session-Id` from an older client ignores it and
must not mint or echo session ids. An open connection — including a
stdio process — is not a session: clients may interleave unrelated
requests on the same transport.

## MCP deprecated Roots, Sampling and Logging

All three client/server features are deprecated in 2026-07-28: they
remain functional during the deprecation window (minimum twelve months)
but new implementations should not add them. Migrations: pass
directories or files via tool parameters, resource URIs, or server
configuration instead of Roots; integrate directly with LLM provider
APIs instead of Sampling; log to stderr on stdio or use OpenTelemetry
instead of the Logging feature.

## Server-initiated requests were replaced by Multi Round-Trip Requests

Servers must not send their own JSON-RPC requests to clients any more —
`roots/list`, `sampling/createMessage` and `elicitation/create` are no
longer delivered as separate requests on any stream. Instead the server
answers the client's request with an `InputRequiredResult`
(`resultType: "input_required"`) whose `inputRequests` map carries what
it needs, and the client retries the original request with
`inputResponses`. This is a breaking change from every revision before
2026-07-28.

## resources/subscribe was replaced by subscriptions/listen

`resources/subscribe`, `resources/unsubscribe` and the standalone HTTP
GET notification stream are gone. A client opens one long-lived
`subscriptions/listen` request whose response stream carries only the
notification types it opted into: `toolsListChanged`,
`promptsListChanged`, `resourcesListChanged`, and `resourceSubscriptions`
(a list of resource URIs). The server tags every notification with
`io.modelcontextprotocol/subscriptionId`.

## MCP removed ping, logging/setLevel and roots list_changed

`ping` is gone. `logging/setLevel` is gone: log level is set per request
via `io.modelcontextprotocol/logLevel` in `_meta`, and servers must not
emit `notifications/message` for a request that did not include that
field. `notifications/roots/list_changed` is gone along with the
deprecated Roots feature.

## SSE stream resumability was removed from Streamable HTTP

`Last-Event-ID` and SSE event ids are gone. A broken response stream
loses the in-flight request; the client must re-issue it as a new
request with a new request id. Servers receiving a `Last-Event-ID`
header from an older client simply ignore it.

## Tasks moved from the core protocol to an extension

Experimental tasks now live in the official extension
`io.modelcontextprotocol/tasks`, negotiated via the `extensions`
capability field. The redesign replaced the blocking `tasks/result` with
polling via `tasks/get`, added `tasks/update` for client-to-server
input, removed `tasks/list`, and lets servers return task handles
unsolicited without per-request opt-in.

## Resource not found changed from -32002 to -32602

Revision 2026-07-28 aligned the resource-not-found error with JSON-RPC:
servers emit `-32602` (Invalid params). Clients should still accept
`-32002` from servers implementing earlier revisions, but
implementations of this revision must not emit it. `-32042` (URL
elicitation required, 2025-11-25 only) is also retired.

## MCP error code allocation policy

JSON-RPC's server-error range is partitioned: `-32000` to `-32019` is
legacy implementation-defined space where new codes must not be
allocated; `-32020` to `-32099` is reserved for the MCP specification.
Spec-defined codes: `-32020` HeaderMismatch, `-32021`
MissingRequiredClientCapability, `-32022` UnsupportedProtocolVersion
(renumbered from -32001, -32003, -32004 in the draft). Application
errors belong outside the JSON-RPC reserved range `-32768` to `-32000`.

## Dynamic Client Registration deprecated for Client ID Metadata Documents

OAuth 2.0 Dynamic Client Registration (RFC 7591) is deprecated as the
MCP client registration mechanism in favour of Client ID Metadata
Documents. DCR remains available for authorization servers that do not
support metadata documents. Related hardening in the same revision:
clients must key persisted credentials by the issuer identifier, must
not reuse them with a different authorization server, and must
re-register when the authorization server changes; authorization servers
should include the `iss` parameter (RFC 9207) and clients must validate
a present `iss` against the recorded issuer before redeeming the code;
clients must specify an appropriate `application_type` during
registration.

## Required _meta fields on every MCP request

Two fields are required on every request, and a request missing either
is malformed — rejected with `-32602` and, on HTTP, status 400:
`io.modelcontextprotocol/protocolVersion` (the revision string) and
`io.modelcontextprotocol/clientCapabilities`. Clients should also send
`io.modelcontextprotocol/clientInfo` (an Implementation object with
name and version) on every request, and servers should identify
themselves with `io.modelcontextprotocol/serverInfo` in every result's
`_meta`. clientInfo and serverInfo are self-reported, unverified, for
display and debugging only — never for security decisions.

## Server rejects capabilities the client did not declare

A server must not rely on capabilities absent from the request's
`io.modelcontextprotocol/clientCapabilities`. If a request needs one the
client did not declare, the server returns
`MissingRequiredClientCapabilityError` (`-32021`) with
`data.requiredCapabilities` listing what was missing; on HTTP the status
is 400. Concretely: no `elicitation/create` in `inputRequests` unless
the client declared elicitation support.

## _meta key naming rules

A `_meta` key is an optional prefix plus a name. The prefix is
dot-separated labels ending in `/`, reverse-DNS style
(`com.example/`); any prefix whose second label is
`modelcontextprotocol` or `mcp` is reserved (so `io.modelcontextprotocol/`
and `dev.mcp/` are reserved, `com.example.mcp/` is not). The name must
begin and end with an alphanumeric. Exception: `traceparent`,
`tracestate` and `baggage` are reserved unprefixed for OpenTelemetry
trace context, following W3C Trace Context and Baggage formats.

## server/discover: mandatory discovery RPC

Servers must implement `server/discover`, which advertises supported
protocol versions, capabilities, and identity. Clients may call it
before any other request for up-front version selection — and on stdio
should use it as the backward-compatibility probe — but are free to
skip it and handle `UnsupportedProtocolVersionError` inline instead.

## UnsupportedProtocolVersionError: the server response for a version it does not implement

There is no negotiation exchange. Every request declares its version;
the server accepts or rejects each request independently. On rejection
the server returns `UnsupportedProtocolVersionError` (`-32022`) whose
`data.supported` lists its versions and `data.requested` echoes the
attempt; the client picks a mutual version from `supported` and retries.

```json
"error": {"code": -32022, "message": "Unsupported protocol version",
  "data": {"supported": ["2026-07-28", "2025-11-25"], "requested": "1900-01-01"}}
```

## Extension negotiation via the extensions capability field

`ClientCapabilities` and `ServerCapabilities` both gained an
`extensions` field: a map from extension identifier (which must follow
`_meta` key naming with a mandatory prefix) to a per-extension settings
object, empty object meaning support with no settings. If one party
supports an extension and the other does not, the supporting party must
revert to core behaviour or reject with an appropriate error. Notable
official extensions: `io.modelcontextprotocol/tasks` (async operations)
and `io.modelcontextprotocol/ui` (MCP Apps inline UI).

```json
"capabilities": {"tools": {},
  "extensions": {"io.modelcontextprotocol/tasks": {}}}
```

## Modern, legacy, and dual-era MCP implementations

The spec names eras: modern revisions (2026-07-28 and later) convey
version, identity and capabilities per request; legacy revisions
(2025-11-25 and earlier) establish a session with `initialize`; a
dual-era implementation supports both. A modern client against a legacy
server fails; a dual-era client detects the server's era — stdio: probe
with `server/discover` and fall back to `initialize` on any error that
is not a recognized modern error; HTTP: attempt a modern request and
inspect the body of a 400 before falling back. Era is a property of the
server: cache it per process or origin, re-probe if the assumption
fails. A dual-era server selects semantics from how the client opens —
per-request `_meta` means modern, `initialize` means legacy.

## JSON-RPC request rules in MCP

Requests carry `jsonrpc: "2.0"`, an id, a method, and optional params.
The id must be a string or integer — unlike base JSON-RPC it must not
be null — and must not match any other in-flight request id from the
same sender. Notifications are one-way messages that must not include
an id and must never receive a response.

## Every MCP result carries resultType

Results are polymorphic: `resultType` is a required string telling the
client how to parse the result. `"complete"` means the final content;
`"input_required"` means an InputRequiredResult asking for more input.
Extensions may add values, but only ones advertised via capabilities; an
unrecognized value is invalid. For backward compatibility, a result from
an earlier-protocol server that omits `resultType` is treated as
`"complete"`.

## Statelessness rules for MCP servers

All information needed to process a request is in the request itself.
Servers must not rely on prior requests over the same connection for
context (capabilities, version, identity); should handle requests from
multiple tasks, threads or conversations interleaved; and should not
require connection reuse for related operations. State spanning requests
must be referenced by an explicit identifier the client passes each
time. A stdio process is not a conversation boundary. Long-lived
requests like `subscriptions/listen` stay request/response — their state
is scoped to the request, not the connection.

## MRTR flow: input_required then retry

The multi round-trip request pattern: (1) client sends a request; (2)
server, needing more information, responds `resultType:
"input_required"` with `inputRequests` and optionally `requestState`;
(3) client gathers the inputs and retries the original request with
`inputResponses` plus the echoed `requestState`, under a NEW JSON-RPC
id; (4) server completes. The retry is completely independent — the
server needs nothing beyond what the retry carries, which is what frees
servers from shared storage and stateful load balancing.

## InputRequests: the map of server needs

`inputRequests` maps server-assigned string keys — unique within the
request — to request objects, each of which must be an ElicitRequest,
CreateMessageRequest, or ListRootsRequest.

```json
"inputRequests": {"github_login": {"method": "elicitation/create",
  "params": {"mode": "form", "message": "Please provide your GitHub username",
    "requestedSchema": {"type": "object",
      "properties": {"name": {"type": "string"}}, "required": ["name"]}}}}
```

## InputResponses: the client's answers on retry

`inputResponses` mirrors the `inputRequests` keys; each value is the
client's result for that request — an ElicitResult (with `action` of
accept/decline/cancel and `content`), CreateMessageResult, or
ListRootsResult. A server receiving extra unrecognized entries should
ignore them; a server missing needed answers should respond with a new
InputRequiredResult asking again rather than erroring.

```json
"inputResponses": {"github_login": {"action": "accept",
  "content": {"name": "octocat"}}}
```

## Only tools/call, resources/read and prompts/get may receive input_required

Only `tools/call`, `resources/read` and `prompts/get` may receive an
InputRequiredResult — servers must not send one on any other request.
Every InputRequiredResult must include at least one of `inputRequests`
or `requestState`. Servers must not include request types the client's
capabilities do not cover, must not assume the client will retry, and
may return input_required repeatedly across attempts until satisfied.

## What a client does with requestState: echo it back exactly, never inspect it

The client's whole obligation for the `requestState` string it receives
in an InputRequiredResult: echo the exact value back when retrying the
original request, never inspect, parse, modify or assume anything about
its contents, and include none if the result carried none. It exists so
the server can reconstitute its own context statelessly on the retry.

## requestState is opaque to clients and hostile to servers

`requestState` is an opaque string only the server understands — any
format (base64 JSON, encrypted JWT, serialized binary). The server must treat
an incoming `requestState` as attacker-controlled: if it influences
authorization, resource access or business logic it must be
integrity-protected (HMAC or AEAD) and rejected on verification
failure. To bound replay, embed and verify the authenticated principal,
a short TTL, and an identifier of the originating request (method plus a
digest of salient parameters). One-time semantics still require
server-side enforcement — integrity protection alone does not give
single-use.

## Streamable HTTP: one POST endpoint

The server exposes a single MCP endpoint accepting POST; every JSON-RPC
request or notification is its own HTTP POST. The client must send
`Accept: application/json, text/event-stream` and the required metadata
headers. The body is a single request or notification — clients never
POST JSON-RPC responses. A notification POST returns `202 Accepted`
with no body on success. A request POST returns either
`application/json` (one object) or `text/event-stream` (an SSE stream
scoped to that request); clients must support both.

## SSE response streams carry only request-scoped messages

On a request's SSE stream the server may send notifications that relate
to that request — `notifications/progress`, `notifications/message` —
before the final response, which should terminate the stream. The
server must not send independent JSON-RPC requests on the stream:
sampling, elicitation and roots ride inside InputRequiredResult per
MRTR. Servers should set `X-Accel-Buffering: no` so reverse proxies
deliver events immediately, and on long-lived streams emit an SSE
comment line (starting with a colon) periodically as keep-alive.

## Cancellation on Streamable HTTP is closing the stream

Closing the SSE response stream is the cancellation signal for that
request — unambiguous because each request has its own stream. The
server should stop work promptly and must not send further messages for
it. `notifications/cancelled` exists only on the stdio transport; the
core protocol defines no client-to-server notifications over Streamable
HTTP.

## MCP-Protocol-Version header on every POST

Every POST to the MCP endpoint carries `MCP-Protocol-Version` matching
the body's `io.modelcontextprotocol/protocolVersion` exactly; a
mismatch is 400 + HeaderMismatch (-32020). Unsupported version: 400 +
UnsupportedProtocolVersionError listing supported versions. Unknown
method: 404 + JSON-RPC -32601, distinguishing a modern server from a
legacy HTTP+SSE 404. A server supporting pre-2025-06-18 clients may
treat a missing header as 2025-03-26; otherwise it rejects.

## Mcp-Method and Mcp-Name request headers

The transport mirrors body fields into headers so load balancers and
gateways can route without parsing the body. `Mcp-Method` (from
`method`) is required on all requests; `Mcp-Name` (from `params.name`
or `params.uri`) is required on `tools/call`, `resources/read` and
`prompts/get`. Header names compare case-insensitively; values are
case-sensitive.

```http
POST /mcp HTTP/1.1
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: get_weather
```

## x-mcp-header: mirroring tool parameters into headers

A tool's inputSchema may annotate a parameter with `x-mcp-header:
"Name"`, and conforming HTTP clients must mirror that argument into an
`Mcp-Param-Name` header. Constraints: value must be RFC 9110 token
syntax, non-empty, no control characters, case-insensitively unique in
the schema; only primitive types (string, integer within the JS safe
range, boolean — `number` is not permitted); only on properties
statically reachable through a chain of `properties` keys — never
through `items`, composition keywords, conditionals, or `$ref`. A
violating annotation invalidates the whole tool definition: HTTP
clients must exclude that tool from `tools/list` and should log why.
Stdio clients may ignore the annotations entirely.

## Non-ASCII values in Mcp-Name and Mcp-Param headers use the base64 sentinel

Header values must be visible ASCII. A value that is not — non-ASCII,
control characters, leading or trailing whitespace — is carried as
`=?base64?{base64-of-UTF-8}?=`, lowercase markers exactly as shown.
This applies to `Mcp-Param-{Name}` and to `Mcp-Name` itself. A plain
value that happens to match the sentinel pattern must also be encoded.
Booleans become lowercase true or false; integers decimal strings.
Servers decode before comparing to the body.

## Header-body validation and HeaderMismatch

Any server that processes the body must validate that mirrored headers
(decoded if base64-encoded) match the corresponding body values, and
reject mismatches, missing required headers, or invalid characters with
400 + `-32020` HeaderMismatch. This closes the vulnerability where a
load balancer routes on the header while the server executes the body.
Integer comparisons are numeric, so 42.0 equals 42. If a mismatch stems
from missing Mcp-Param headers, the client should refresh `tools/list`
(the schema may have changed) and retry. Intermediaries that rate-limit
or route on mirrored headers should verify the protocol version is one
that requires validation before trusting them.

## Detecting a legacy server over HTTP

A dual-era client attempts a modern request first. On 400 it inspects
the body: a recognized modern JSON-RPC error
(UnsupportedProtocolVersion, MissingRequiredClientCapability,
HeaderMismatch) proves a modern server — retry with an advertised
version rather than falling back. An empty or unrecognized body means
legacy — fall back to `initialize`. For the 2024-11-05 HTTP+SSE era, a
failed POST is followed by GET expecting an `endpoint` SSE event. A
modern-only server answers stray GET or DELETE with 405, and should
name its supported versions in any error it returns to an `initialize`
attempt — legacy clients have no fall-forward and that message may be
their only diagnostic.

## subscriptions/listen request shape

The client opts into notification types with a `notifications` filter;
all fields optional, omission means not subscribed. The server must not
send types the client did not request.

```json
{"method": "subscriptions/listen", "params": {"notifications": {
  "toolsListChanged": true,
  "resourceSubscriptions": ["file:///project/config.json"]}}}
```

`toolsListChanged`, `promptsListChanged`, `resourcesListChanged` are
booleans; `resourceSubscriptions` lists resource URIs for
`notifications/resources/updated`.

## Subscription acknowledgment comes first

The server must send `notifications/subscriptions/acknowledged` as the
first message on a subscription — before any other notification for
that subscription id — carrying the honored subset of the requested
filter (unsupported types omitted). The client should compare the
acknowledged filter to what it asked for. On stdio the ordering is per
subscription id, not per channel, so other subscriptions' messages may
interleave.

## subscriptionId correlates every stream message

Every notification on a listen stream carries
`io.modelcontextprotocol/subscriptionId` in `_meta`, whose value is the
JSON-RPC id of the originating `subscriptions/listen` request. A client
may hold multiple concurrent subscriptions and must demultiplex by this
field on stdio, where everything shares one channel.

## Ending a subscription gracefully

A subscription ends when the client closes the SSE stream (HTTP) or
sends `notifications/cancelled` with the listen request id (stdio);
when the server tears it down; or when the transport drops. A server
ending it deliberately should first answer the original listen request
with an empty `resultType: "complete"` result carrying the
subscriptionId — the graceful-closure signal. A transport that closes
without that response is an unexpected disconnect the client may treat
as a reconnect trigger. After a stdio reconnect the client must re-send
`subscriptions/listen`; the server holds no subscription state.

## CacheableResult: ttlMs and cacheScope on list results

Results of `tools/list`, `prompts/list`, `resources/list`,
`resources/read` and `resources/templates/list` must carry `ttlMs` (a
freshness hint in milliseconds enabling client caching instead of
polling) and `cacheScope` — `"public"` allows shared intermediaries to
cache, `"private"` restricts to the requesting client. Both complement
listChanged notifications. Servers should also return tools in
deterministic order, which enables client-side caching and improves LLM
prompt cache hit rates.

## Tool schemas allow full JSON Schema 2020-12

`inputSchema` and `outputSchema` accept any JSON Schema 2020-12
keywords, and `structuredContent` any JSON value. Without `$schema` the
dialect defaults to 2020-12; implementations must support 2020-12 and
handle unsupported dialects with a clear error. Implementations must
not automatically dereference network `$ref` URIs — an opt-in fetch
mode must default off, allowlist hosts, reject loopback and private
addresses, and apply timeouts and size limits; a schema failing on an
unresolved external `$ref` is rejected, not treated as permissive.
Validators should bound composition keywords (`anyOf`, `oneOf`,
`allOf`, `if`/`then`/`else`, `$defs`) by depth, subschema count, or
time budget so a malicious schema cannot act as denial of service.

## OpenTelemetry trace context in _meta

`traceparent`, `tracestate` and `baggage` ride unprefixed in `_meta`,
following W3C Trace Context and W3C Baggage formats, matching the
OpenTelemetry semantic conventions for MCP.

```json
"_meta": {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01"}
```

## Icon security requirements

Icons attach to Implementation, Tool, Prompt and Resource objects as
`{src, mimeType, sizes, theme}`. Clients rendering them must support
PNG and JPEG, should support SVG and WebP, and must treat icon bytes as
untrusted: HTTPS or `data:` URIs only, reject `javascript:`, `file:`
and other unsafe schemes and cross-origin redirects, fetch without
credentials, verify same-origin with the server, detect content type by
magic bytes and reject mismatches, and guard against oversized images.
SVG may contain embedded JavaScript — sanitize or disallow.

## MCP security principles

Users must explicitly consent to and control all data access and
operations; hosts must obtain consent before exposing user data to
servers and must not transmit resource data elsewhere without it. Tools
are arbitrary code execution: tool descriptions and annotations are
untrusted unless from a trusted server, and hosts must obtain explicit
user consent before invoking any tool. The protocol cannot enforce
these; implementors build the consent flows, access controls, and
documentation.

## MCP hosts, clients and servers

A host is the LLM application initiating connections; a client is the
connector inside the host; a server provides context and capabilities.
Servers offer resources (context and data), prompts (templated
messages), and tools (functions the model executes); clients may offer
elicitation (server-requested user input via MRTR). MCP messages are
JSON-RPC 2.0, inspired by the Language Server Protocol's approach to
ecosystem standardization.

## Which revision string to send

The current protocol revision identifier is the string `2026-07-28`,
sent in `io.modelcontextprotocol/protocolVersion` on every request and
in the `MCP-Protocol-Version` HTTP header. The authoritative schema is
the TypeScript file `schema/2026-07-28/schema.ts` in the
modelcontextprotocol/specification repository, with a generated JSON
Schema alongside; the schema, not the prose, is the source of truth.
Since this revision, minimum/maximum/default in schema.json are
`number`, not `integer`.
