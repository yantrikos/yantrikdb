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

## Which three MCP features were deprecated together in revision 2026-07-28: Roots, Sampling and Logging

The three features deprecated together in MCP revision 2026-07-28 are **Roots**, **Sampling** and **Logging**.

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
Three spec-defined codes, each with its OWN trigger — they are not
interchangeable and the trigger is what selects between them:

- `-32020` HeaderMismatch — the `MCP-Protocol-Version` header does not
  match the version in the request body.
- `-32021` MissingRequiredClientCapability — the client did not declare
  a capability the operation requires.
- `-32022` UnsupportedProtocolVersion — **the version itself is one the
  server does not implement.** This is the code for an unsupported
  version, and its `data.supported` lists the versions the server does
  implement so the client can retry.

(All three were renumbered from -32001, -32003 and -32004 in the draft.)
Application errors belong outside the JSON-RPC reserved range `-32768`
to `-32000`.

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

## inputRequests: how a server delivers elicitation/create or sampling/createMessage to a client

A server never sends these as JSON-RPC requests, because server-initiated requests no longer exist. It returns an `InputRequiredResult`, and **the requests themselves travel in that result's `inputRequests` field** — that field is the delivery mechanism, and naming `InputRequiredResult` without naming `inputRequests` describes the envelope and omits the contents.

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
UnsupportedProtocolVersionError listing supported versions.

At an MCP endpoint specifically — this is MCP's use of the status code,
not HTTP's general meaning of it — an unknown METHOD in the JSON-RPC
body returns 404 together with JSON-RPC -32601. The pairing is what
distinguishes a modern server from a legacy HTTP+SSE server's 404,
which carries no JSON-RPC error at all. In plain HTTP, 404 still means
what it always has: the requested resource does not exist.

A server supporting pre-2025-06-18 clients may treat a missing header
as 2025-03-26; otherwise it rejects.

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

## How a client carries a non-ASCII value in an Mcp-Param or Mcp-Name header: the base64 sentinel

To carry a non-ASCII value in an `Mcp-Param` or `Mcp-Name` header, a client wraps it in the base64 sentinel `=?base64?{base64-of-UTF-8}?=`.

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


## tools/list request and result shape

Clients discover the tools a server exposes by sending the JSON-RPC method `tools/list`. The request takes an optional `cursor` string in `params` for pagination; the result carries a `tools` array, an optional `nextCursor` for the following page, and — new in the 2026-07-28 revision — a `resultType` of `"complete"`, plus the cache fields `ttlMs` and `cacheScope`. Each entry of `tools` is a tool definition object with `name`, optional `title`, `description`, `inputSchema`, and optionally `outputSchema`, `icons`, and `annotations`. Only servers that declare the `tools` capability are required to answer `tools/list`.

## tools/call request and result shape

To invoke a tool a client sends `tools/call` with `params.name` set to the tool's `name` and `params.arguments` set to an object of argument values matching the tool's `inputSchema`. A successful result contains `resultType: "complete"`, a `content` array of content blocks, and `isError` (`false` on success). Results may also carry `structuredContent`. The `resultType` field is new in this revision — code written against earlier MCP revisions emits a `CallToolResult` with only `content` and `isError` and will not populate it.

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call",
 "params": {"name": "get_weather", "arguments": {"location": "New York"}}}
```

## resultType discriminator on every tool result

Every result in this revision carries a `resultType` discriminator. A finished tool call returns `resultType: "complete"`; a call that needs more information from the user returns `resultType: "input_required"`. The same field appears on `tools/list` results, where it is `"complete"`. This is a change from earlier MCP revisions, where `CallToolResult` and `ListToolsResult` had no type tag at all, so clients ported from 2025-era code ignore `resultType` and mis-handle an `input_required` result as a completed call with no content.

## Caching a tools/list result with ttlMs and cacheScope

The result of `tools/list` may include `ttlMs` and `cacheScope`, which let a server say how long the tool list may be cached and how widely. In the specification's example `ttlMs` is `300000` and `cacheScope` is `"public"`. These fields did not exist in earlier revisions. They pair with the requirement that servers return tools in a deterministic order, since a stable list is what makes both client-side caching and LLM prompt caching pay off.

## Declaring the tools server capability with listChanged

A server that exposes tools MUST declare the `tools` capability. The declaration nests under `capabilities`, and its one documented member is the boolean `listChanged`, indicating whether the server will emit notifications when the set of available tools changes. Declaring `tools` obliges the server to respond to `tools/list` with the tools currently available to the requesting client; that set MAY be empty.

```json
{"capabilities": {"tools": {"listChanged": true}}}
```

## Required _meta fields on every tools request

Every request in this revision, including `tools/list` and `tools/call`, MUST include the required `_meta` request metadata: `io.modelcontextprotocol/protocolVersion`, `io.modelcontextprotocol/clientInfo`, and `io.modelcontextprotocol/clientCapabilities`. The examples on the tools page omit `_meta` only for brevity — omitting it on the wire is not conformant. This is the per-request replacement for what earlier revisions exchanged once during the `initialize` handshake, so a client that sends only `name` and `arguments` in `params` and nothing in `_meta` is emitting a malformed 2026-07-28 request.

## The tool list must not vary per connection but may vary by authorization

The set of tools returned by `tools/list` MAY be empty and MAY change over time, but it MUST NOT vary per-connection or as a side effect of other requests on the connection. It MAY vary by the authorization presented on the request — returning only the tools the caller's granted scopes permit — because credentials are per-request input, not connection state. This is a direct consequence of MCP no longer having sessions: a server cannot expose one tool list to a "logged-in" connection and another to a fresh one, and any design that flips tools on and off through a prior tool call on the same connection is non-conformant.

## Deterministic ordering of tools in tools/list

Servers SHOULD return tools in a deterministic order — the same ordering across requests whenever the underlying set has not changed. The stated reasons are that deterministic ordering lets clients reliably cache the tool list, and that it improves LLM prompt cache hit rates when the tools are serialized into model context. A server that builds its `tools` array by iterating an unordered map or set silently defeats both caches even though every individual response is valid.

## notifications/tools/list_changed requires an open subscriptions/listen stream

When the set of available tools changes, a server that declared `listChanged: true` SHOULD send `notifications/tools/list_changed`, which has a `method` and no `params`. In this revision the notification goes only to clients that have opened a `subscriptions/listen` stream with `toolsListChanged: true`; the server acknowledges that stream with `notifications/subscriptions/acknowledged` before change notifications flow. This is a change from earlier revisions, where declaring `listChanged` was enough and the notification was pushed to any connected client. On receiving it, the client re-issues `tools/list`.

## Fields of a tool definition object

A tool definition includes `name`, a unique identifier; `title`, an optional human-readable name for display; `description`, a human-readable description of functionality; `icons`, an optional array of icons for user interfaces; `inputSchema`, a JSON Schema defining expected parameters; `outputSchema`, an optional JSON Schema defining expected output structure; and `annotations`, optional properties describing tool behavior. Each entry in `icons` carries `src`, `mimeType`, and a `sizes` array such as `["48x48"]`. Both `title` and `icons` are display-layer fields — the model selects a tool by `name`.

## Tool annotations are untrusted metadata

The `annotations` member of a tool definition holds optional properties describing tool behavior. For trust and safety, clients MUST consider tool annotations untrusted unless they come from trusted servers — an annotation claiming a tool is harmless is an assertion by the server, not a guarantee the client can rely on when deciding whether to auto-approve a call. The tools page defines `annotations` only at this level and does not enumerate individual annotation field names or defaults, so specific hint names must come from the schema reference.

## Allowed characters and length for tool names

Tool names SHOULD be between 1 and 128 characters inclusive and SHOULD be treated as case-sensitive. The only characters that SHOULD be allowed are uppercase and lowercase ASCII letters, digits, underscore, hyphen, and dot; names SHOULD NOT contain spaces, commas, or other special characters. Names SHOULD be unique within a server. The specification's examples of valid names are `getUser`, `DATA_EXPORT_v2`, and `admin.tools.list`.

## Tool name collisions when aggregating multiple MCP servers

Tool name uniqueness is scoped to a single server. A client or proxy aggregating tools from several servers MAY encounter collisions — two servers each exposing a `search` tool — and SHOULD implement a disambiguation strategy such as prefixing tool names with a server identifier. The specification explicitly warns that the server `name` from `serverInfo` is not guaranteed unique across servers and SHOULD NOT be relied upon for that disambiguation, so an aggregator needs its own identifier rather than trusting what the server calls itself.

## x-mcp-header mirroring a tool parameter into an HTTP header

New in the 2026-07-28 revision, the `x-mcp-header` extension property lets a server designate tool parameters to be mirrored into HTTP headers on the Streamable HTTP transport, so load balancers, proxies, and WAFs can route on parameter values without parsing the body. The property goes directly inside the JSON Schema of the property to be mirrored, and its value gives the name portion of the resulting `Mcp-Param-{name}` header. In the `execute_sql` example, the `region` property carries `"x-mcp-header": "Region"`, so calling with `"region": "us-west1"` makes the client add `Mcp-Param-Region: us-west1`.

```json
"region": {"type": "string", "description": "The region to execute the query in",
           "x-mcp-header": "Region"}
```

## Constraints on x-mcp-header values and client rejection of invalid tools

An `x-mcp-header` value MUST NOT be empty; MUST match HTTP field-name token syntax (`1*tchar`, RFC 9110 Section 5.1); MUST NOT contain control characters including CR or LF; MUST be case-insensitively unique among all `x-mcp-header` values in the same `inputSchema`; MUST only be applied to parameters of primitive type (integer, string, boolean — `number` is not permitted, integers confined to the IEEE754 double safe range); and MUST only be applied to properties statically reachable from the schema root. Clients on Streamable HTTP MUST reject tool definitions that violate any of these, where rejection means excluding that one tool from `tools/list` so a single malformed definition does not disable the others, and SHOULD log a warning naming the tool and reason. Clients on other transports such as stdio MAY ignore `x-mcp-header` entirely.

## Do not put secrets in x-mcp-header parameters

Server developers SHOULD NOT mark sensitive parameters — passwords, API keys, tokens, or PII — with `x-mcp-header`, because the resulting `Mcp-Param-{name}` header values are visible to the network intermediaries the mechanism exists to serve. The whole point is that load balancers, proxies, and WAFs read the value without parsing the body, so anything mirrored into a header should be treated as disclosed to that infrastructure and to anything that logs it.

## inputSchema for a tool that takes no parameters

`inputSchema` MUST be a valid JSON Schema object and MUST NOT be `null`. For a tool with no parameters the specification gives two valid forms: `{"type": "object", "additionalProperties": false}`, which is recommended because it explicitly accepts only empty objects, and `{"type": "object"}`, which accepts any object including one with properties. Omitting `inputSchema` or setting it to `null` is not an option.

## JSON Schema dialect for tool inputSchema and outputSchema

Both `inputSchema` and `outputSchema` follow the JSON Schema usage guidelines from the basic specification, and both default to JSON Schema 2020-12 when no `$schema` field is present. A tool may pin an older dialect explicitly — the specification shows a `calculate_sum` tool whose `inputSchema` sets `"$schema": "http://json-schema.org/draft-07/schema#"`. When validating arguments and results against these schemas, clients SHOULD follow the `$ref` resolution requirements defined in the basic specification.

## structuredContent may be any JSON value, not just an object

Structured tool output is returned in the `structuredContent` field, and in this revision it can be any JSON value — object, array, string, number, boolean, or null — conforming to the tool's `outputSchema` if one is defined. This widens earlier revisions, where structured content was an object; a `list_users` tool declaring an `outputSchema` of `{"type": "array", "items": {...}}` may return a bare JSON array. For backwards compatibility a tool returning structured content SHOULD also return the serialized JSON in a TextContent block. Note that `structuredContent` is server-produced result data, unrelated to LLM "structured outputs" meaning schema-constrained model generation.

## outputSchema validation duties for servers and clients

If a tool provides an `outputSchema`, servers MUST provide structured results conforming to it and clients SHOULD validate structured results against it. The stated benefits are strict schema validation of responses, type information for better integration with programming languages, better parsing by clients and LLMs, and improved documentation. A `get_weather_data` tool declaring `temperature`, `conditions`, and `humidity` as required must populate all three in `structuredContent`, not merely describe them in the `content` text.

## Content block types allowed in a tool result

Unstructured tool output goes in the result's `content` array, which may hold multiple items of differing types: `text` (with a `text` string), `image` (base64 `data` plus a `mimeType` such as `image/png`), `audio` (base64 `data` plus a `mimeType` such as `audio/wav`), `resource_link`, and embedded `resource`. All five support optional `annotations` carrying metadata about audience, priority, and modification times — the same annotation format used by resources and prompts, with fields such as `audience: ["user", "assistant"]`, `priority: 0.7`, and `lastModified`.

## Returning a resource_link versus an embedded resource from a tool

A tool MAY return a `resource_link` content block — with `uri`, `name`, `description`, and `mimeType` — pointing the client at a resource it can fetch or subscribe to, rather than inlining the bytes. Resource links returned by tools are not guaranteed to appear in `resources/list`, so a client must not assume it can rediscover the URI through listing. Alternatively a tool MAY embed the resource directly as a content block of `type: "resource"` whose `resource` object carries `uri`, `mimeType`, and `text`; servers using embedded resources SHOULD implement the `resources` capability.

## Protocol errors versus tool execution errors

Tools report failures two distinct ways. Protocol errors cover problems with the request structure a model is unlikely to fix — an unknown tool, a malformed request failing the `CallToolRequest` schema, or a server error — and are returned as standard JSON-RPC errors; the specification's example uses code `-32602` with message `Unknown tool: invalid_tool_name`. Tool execution errors cover API failures, input validation errors such as a wrongly formatted date or an out-of-range value, and business logic errors; these are returned as a normal successful JSON-RPC result carrying `isError: true` with an explanatory text content block. Clients SHOULD provide tool execution errors to language models so they can self-correct and retry, and MAY provide protocol errors, though those are less likely to lead to recovery.

## tools/call returning input_required and the retry rules

A server MAY answer `tools/call` with an `InputRequiredResult` instead of finishing, using `resultType: "input_required"`. The result carries an `inputRequests` object keyed by a server-chosen name such as `github_login`, whose value holds a `method` (for example `elicitation/create`) and its `params`, and it may carry an opaque `requestState` string. The client retries by re-sending `tools/call` with the same `name` and `arguments`, an `inputResponses` object keyed identically — each entry having an `action` such as `"accept"` and a `content` object — and the `requestState` echoed back if supplied. The JSON-RPC `id` MUST differ between the initial request and the retry.

## Keeping state across tool calls with an explicit handle

MCP has no protocol-level session in this revision, so a server cannot rely on implicit per-connection state to relate one tool call to the next. The non-normative guidance is that a server needing state across calls — a shopping cart, an open browser context, a database transaction — should return an explicit handle from a creation tool and accept it as an argument on subsequent calls: a `create_basket` tool returns `structuredContent` of `{"basket_id": "bsk_a1b2c3"}`, and `add_item` takes `basket_id` back as an ordinary argument. The protocol has no concept of a state handle; from the wire's perspective it is a string in a result and a string in the next call's `arguments`.

## Designing tool state handles: authorization, opacity, lifetime, expiry

When a server hands out state handles, four things matter. Authorization: for authenticated servers a handle is a name, not a capability, so the server should validate the caller's authorization against it on every call; for unauthenticated servers the handle is necessarily a bearer token and should be generated with sufficient entropy such as a UUIDv4 and given a bounded lifetime. Opacity: handles encoding internal structure invite parsing or guessing, so opaque identifiers are preferred. Lifetime: because handles outlive any connection, the retention policy should be stated in the creation tool's `description` — "baskets expire after 24 hours of inactivity" — so the model sees it. Expiry: a call against an expired or unknown handle should return a tool execution error saying so, letting the model create a new one.

## Security requirements for servers exposing tools and clients calling them

Servers MUST validate all tool inputs, implement proper access controls, rate limit tool invocations, and sanitize tool outputs. Clients SHOULD prompt for user confirmation on sensitive operations; show tool inputs to the user before calling the server, to avoid malicious or accidental data exfiltration; validate tool results before passing them to the LLM; follow the `$ref` resolution requirements when validating against `inputSchema` and `outputSchema`; implement timeouts for tool calls; and log tool usage for audit purposes.

## Human in the loop for MCP tool invocation

Tools in MCP are model-controlled, meaning the language model can discover and invoke them automatically from contextual understanding and user prompts, and the protocol mandates no particular user interaction model. For trust, safety, and security there SHOULD nonetheless always be a human in the loop able to deny tool invocations. Applications SHOULD provide UI making clear which tools are exposed to the model, insert visual indicators when tools are invoked, and present confirmation prompts so a human really is in the loop.

## The three MCP server primitives and who controls each

A server contributes context to a language model through three primitives. Prompts are pre-defined templates or instructions guiding model interactions; they are user-controlled, invoked by user choice as with slash commands. Resources are structured data or content providing additional context; they are application-controlled, attached and managed by the client, as with file contents or git history. Tools are executable functions letting the model perform actions or retrieve information; they are model-controlled, exposed to the LLM to take actions such as API requests or file writes. This control hierarchy decides which primitive a capability belongs in: if the model should decide when to fire it, it is a tool, not a resource.


## resources/list request and the ListResourcesResult fields

Clients discover resources by sending `resources/list`, whose `params` may carry an optional `cursor` string for pagination. The result contains `resultType` (`"complete"` ordinarily), a `resources` array, an optional `nextCursor` when more pages remain, and the caching fields `ttlMs` and `cacheScope`. Each entry carries `uri`, `name`, and optionally `title`, `description`, `mimeType`, and `icons`. The `resultType`, `ttlMs`, and `cacheScope` fields are new in this revision and did not appear in earlier MCP list results. Like every request here, `resources/list` must include the required `_meta` fields, which the spec's examples omit only for brevity.

## resources/read request and the contents result array

To fetch the body of a resource a client sends `resources/read` with a single required `params.uri`, for example `file:///project/src/main.rs`. The result contains `resultType`, a `contents` array, and the caching fields. Every entry in `contents` repeats the `uri` it came from, carries a `mimeType`, and holds the payload in exactly one of two fields: `text` for textual data or `blob` for base64-encoded binary. Servers MAY return multiple entries for a single read — for example the contents of several files when a directory resource is read — so clients must not assume a one-to-one mapping between the requested `uri` and the returned entries. `resources/read` supports caching but not pagination.

```json
{"resultType": "complete",
 "contents": [{"uri": "file:///project/src/main.rs",
               "mimeType": "text/x-rust", "text": "fn main() {}"}],
 "ttlMs": 60000, "cacheScope": "private"}
```

## resources/read and prompts/get may return InputRequiredResult

Both `resources/read` and `prompts/get` MAY be answered with an `InputRequiredResult` rather than the normal result, signalling that the server needs additional input before it can read the resource or resolve the prompt. When the client retries, it includes `inputResponses` in the request `params`, plus `requestState` if the server supplied one. This is new in 2026-07-28: in earlier revisions these two methods could only return their content result or a JSON-RPC error, so client code written against older specs mis-handles an `InputRequiredResult` as an unexpected payload.

## resources/templates/list and the uriTemplate field for parameterized resources

Servers expose parameterized resources through `resources/templates/list`, which accepts an optional `cursor` and returns `resultType`, a `resourceTemplates` array, `nextCursor`, `ttlMs`, and `cacheScope`. Each template entry uses `uriTemplate` — an RFC 6570 URI template such as `file:///{path}` — in place of the concrete `uri` a plain resource carries, alongside `name`, optional `title`, `description`, `mimeType`, and `icons`. Template arguments may be auto-completed through `completion/complete`, so a client can suggest values for `{path}` before expanding the template into a concrete URI to pass to `resources/read`. Note the method name is `resources/templates/list`, not `resources/list/templates`.

## Fields of a Resource definition: uri, name, title, description, icons, mimeType, size

A resource definition includes `uri`, the unique identifier; `name`, the programmatic name; `title`, an optional human-readable name for display; `description`, optional; `icons`, an optional array for user interfaces; `mimeType`, optional; and `size`, an optional size in bytes. The `icons` entries carry `src`, `mimeType` (for example `image/png` or `image/svg+xml`), and `sizes` (`["48x48"]` or `["any"]`). Both `title` and `icons` are display-oriented additions relative to the original `name`/`description`-only shape, so a client reading only `name` shows the raw identifier where a friendlier `title` was intended.

## Resource annotations: audience, priority, and lastModified hints

Resources, resource templates, and content blocks all support an optional `annotations` object giving clients hints about how to use or display the item. `audience` is an array with the valid values `"user"` and `"assistant"`. `priority` is a number from 0.0 to 1.0 where 1 means most important (effectively required) and 0 means least important (entirely optional). `lastModified` is an ISO 8601 timestamp such as `"2025-01-12T15:00:58Z"`. Clients use these to filter by intended audience, prioritize what enters the model's context, and sort by recency. The same annotations apply to prompt-message content blocks and to `resource_link` blocks.

## Declaring the resources capability with listChanged and subscribe

A server supporting resources MUST declare the `resources` capability, which has two optional boolean sub-capabilities. `listChanged` states whether the server emits notifications when the set of resources changes; `subscribe` states whether it supports resource-specific update notifications for resources requested through `subscriptions/listen` using the `resourceSubscriptions` filter. A server supporting neither may declare simply `{"capabilities": {"resources": {}}}`. When the list changes, a server that declared `listChanged` SHOULD send `notifications/resources/list_changed`, which carries no `params`. Note that in this revision `subscribe` no longer means the server implements a `resources/subscribe` method — it means the server honours the `resourceSubscriptions` filter on a listen stream.

## resources/subscribe was removed: watching a resource URI in 2026-07-28

The `resources/subscribe` and `resources/unsubscribe` methods no longer exist. To watch specific resources, a client sends `subscriptions/listen` listing the URIs it cares about in `notifications.resourceSubscriptions`; the server then delivers `notifications/resources/updated` on the resulting stream whenever a watched resource changes. The notification's `params` carry the changed `uri` plus `_meta` holding `io.modelcontextprotocol/subscriptionId`. This is the largest behavioural change on the resources page — models trained on earlier revisions emit a `resources/subscribe` request with `{"uri": ...}`, which a 2026-07-28 server rejects as an unknown method.

```json
{"jsonrpc": "2.0", "method": "notifications/resources/updated",
 "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": 4},
            "uri": "file:///project/src/main.rs"}}
```

## Error code for a resource that does not exist is -32602, not -32002

If the requested resource does not exist, servers MUST return a JSON-RPC error with code `-32602` (Invalid Params), and SHOULD return `-32603` for internal errors. For backwards compatibility clients SHOULD also accept `-32002`, which earlier protocol versions used — but a server written to this revision must emit `-32602`. Servers MUST NOT return an empty `contents` array for a non-existent resource: an empty array is ambiguous, since it could mean the resource exists but has no content. The error object's `data` field echoes the offending URI back to the client.

## When to use the https:// URI scheme for an MCP resource

The `https://` scheme represents a resource available on the web. Servers SHOULD use it only when the client can fetch and load the resource directly from the web on its own — that is, when the client does not need to read it through the MCP server at all. For every other case servers SHOULD prefer another scheme, or define a custom one, even when the server itself downloads the contents over the internet. The rule is about who does the fetching, not where the bytes live. Correspondingly, when a returned `uri` uses `https://`, a client may skip `resources/read` and fetch directly.

## file://, git://, and custom URI schemes for MCP resources

The protocol names several standard URI schemes and the list is explicitly not exhaustive — implementations may use additional custom schemes. The `file://` scheme identifies resources behaving like a filesystem, though they need not map to any physical filesystem; servers MAY identify `file://` resources with an XDG shared-MIME-info type such as `inode/directory` to represent non-regular files like directories that have no other standard MIME type. The `git://` scheme is defined for Git version control integration. Any custom scheme MUST conform to RFC 3986 and take the `https://` and `file://` guidance into account.

## Security requirements for serving MCP resources

Servers MUST validate all resource URIs. Access controls SHOULD be implemented for sensitive resources. Binary data MUST be properly encoded — the `blob` field must be genuine base64. Resource permissions SHOULD be checked before operations. And servers MUST sanitize file paths to prevent directory traversal attacks when serving `file://` resources, which matters especially for templates like `file:///{path}` where the client supplies the expansion.

## Why resources/list and prompts/list results must not vary per connection

Both pages state the same stability rule. A server declaring `resources` MUST respond to `resources/list` with the set currently available to the requesting client, and likewise for `prompts`. That set MAY be empty and MAY change over time (signalled by list-changed notifications) but MUST NOT vary per-connection or as a side effect of other requests on the connection. It MAY vary by the authorization presented on the request, because credentials are per-request input, not connection state. The rule exists because this revision has no initialize handshake and no session in which per-connection state could legitimately accumulate.

## prompts/list request and the ListPromptsResult fields

Clients retrieve available prompts with `prompts/list`, taking an optional `cursor`. The result carries `resultType`, a `prompts` array, an optional `nextCursor`, and the caching fields `ttlMs` and `cacheScope`. Each prompt entry holds `name` (for example `code_review`), optional `title` (`"Request Code Review"`), optional `description`, an optional `arguments` array, and an optional `icons` array. The `resultType`, `ttlMs`, and `cacheScope` fields are new in 2026-07-28 — earlier revisions returned only `prompts` and `nextCursor`.

## prompts/get request with name and arguments, and the messages result

To retrieve a specific prompt a client sends `prompts/get` with `params.name` set to the prompt's `name` and `params.arguments` set to an object of argument name/value string pairs. Note that `arguments` on the request is a map, not the array of argument descriptors returned by `prompts/list`. The result contains `resultType`, an optional `description` describing the resolved prompt, and a `messages` array of `PromptMessage` objects each with a `role` and a `content` block. Argument values may be auto-completed through `completion/complete` before the client issues `prompts/get`. Unlike `prompts/list`, this operation is not paginated.

## Fields of a Prompt definition and its PromptArgument entries

A prompt definition includes `name`, the unique identifier; `title`, an optional human-readable name for display; `description`, optional, such as `"Asks the LLM to analyze code quality and suggest improvements"`; `icons`, optional; and `arguments`, an optional list for customization. Each entry of `arguments` is an object with `name` (the key the client sends in `prompts/get`), `description`, and `required` (a boolean). A client rendering prompts as slash commands should prefer `title` over the machine-facing `name`.

## PromptMessage role field and the text, image, and audio content blocks

Each entry of the `messages` array has a `role` of either `"user"` or `"assistant"` and a `content` object discriminated by its `type`. A text block is `{"type": "text", "text": "..."}` and is the most common for natural-language interaction. An image block is `{"type": "image", "data": "...", "mimeType": "image/png"}`, where data MUST be base64-encoded with a valid MIME type. An audio block is `{"type": "audio", "data": "...", "mimeType": "audio/wav"}` under the same encoding rule. All content types in prompt messages also support the optional `annotations` object carrying `audience`, `priority`, and `lastModified`.

## resource_link content blocks in prompt messages

A prompt message MAY link to a resource instead of embedding its bytes. The block has `type` set to `"resource_link"` and carries the resource's `uri`, `name`, `description`, and `mimeType` directly on the block itself — there is no nested `resource` object, which is what distinguishes it from an embedded resource. The client is expected to fetch the URI itself, typically via `resources/read`. Resource links support the same annotations as regular resources, so a client can decide whether the linked material is worth pulling into context.

## Embedded resource content blocks in prompt messages

An embedded resource inlines server-side resource content directly into a prompt message. The block has `type` set to `"resource"` and a nested `resource` object — this nesting is the difference from `resource_link`, where the fields sit on the block itself. The nested object holds either text or binary data and MUST include a valid resource URI, the appropriate MIME type, and either `text` content or base64-encoded `blob` data. Embedded resources let prompts incorporate server-managed content such as documentation or code samples without a second round trip.

```json
{"type": "resource",
 "resource": {"uri": "resource://example", "mimeType": "text/plain",
              "text": "Resource content"}}
```

## Declaring the prompts capability and delivering notifications/prompts/list_changed

Servers supporting prompts MUST declare the `prompts` capability, whose single sub-capability `listChanged` indicates whether the server emits notifications when the prompt list changes. When it does change, a server that declared `listChanged` SHOULD send `notifications/prompts/list_changed` — but only to clients that have opened a `subscriptions/listen` stream with `promptsListChanged: true`. That delivery precondition is new in 2026-07-28: earlier revisions pushed the notification to any connected client that had negotiated the capability. Note also that the capability is declared in a `DiscoverResult`, not in an initialize response.

## Error codes returned by prompts/list and prompts/get

Servers SHOULD return standard JSON-RPC errors for the common prompt failures. An invalid prompt name — a `prompts/get` naming a prompt the server does not have — is `-32602` (Invalid params). Missing required arguments, meaning the request omitted an argument whose `required` flag is true, is also `-32602`. Internal errors are `-32603`. Unlike the resources page, the prompts page states these as SHOULD rather than MUST and defines no prompt-specific code, so `-32602` covers both name and argument problems.

## Implementation and security requirements for MCP prompt servers

Servers SHOULD validate prompt arguments before processing them. Clients SHOULD handle pagination for large prompt lists, following `nextCursor`. Both parties SHOULD respect capability negotiation, so a client should not call `prompts/list` or `prompts/get` against a server that did not declare the `prompts` capability. For security, implementations MUST carefully validate all prompt inputs and outputs to prevent injection attacks or unauthorized resource access — covering both the `arguments` a client supplies and the `messages` content a server returns, including embedded resources and resource links.


## server/discover request params and required _meta fields

The `server/discover` request carries no body parameters of its own — everything travels in the standard `_meta` object on `params`. The three keys are `io.modelcontextprotocol/protocolVersion` (the string `"2026-07-28"`), `io.modelcontextprotocol/clientInfo` (an object with `name` and `version`), and `io.modelcontextprotocol/clientCapabilities` (an object, may be empty). The method string is exactly `server/discover` and it is a normal JSON-RPC request with an `id`. Because the body is empty, everything the server learns about the client on a discover call comes from `_meta`, not from named parameters.

## DiscoverResult response fields returned by server/discover

The `DiscoverResult` contains `resultType` (`"complete"`), `supportedVersions`, `capabilities`, `instructions`, and the caching hints `ttlMs` and `cacheScope`, plus a `_meta` object. `supportedVersions` is an array of protocol version strings such as `["2026-07-28"]`, and the client should choose one for subsequent requests. `capabilities` is the object of capabilities the server supports — `tools`, `resources`, `prompts` and so on — so a client can render everything a server offers from this one response instead of probing with separate list calls. `instructions` is optional natural-language guidance for LLMs on how to use the server effectively.

## Where serverInfo lives in a DiscoverResult and whether clients may trust it

The server's name and version are not top-level fields of `DiscoverResult` — they live under `_meta` at the key `io.modelcontextprotocol/serverInfo`, an object with `name` and `version`, which servers SHOULD include. The specification is explicit that `serverInfo` is self-reported and not verified by the protocol: it is intended for display, logging, and debugging only. Clients SHOULD NOT use it to change behavior and SHOULD NOT rely on it for security decisions. Code that keys behavior off a server's reported name or version is doing something the spec warns against.

## Is calling server/discover required before other MCP requests

Servers MUST implement `server/discover`, but calling it is optional for clients. A client may invoke any RPC inline and handle `UnsupportedProtocolVersionError` if the server does not support the version it requested. The spec names two scenarios where calling it is nonetheless useful: presenting server information (identity, capabilities, and supported versions in a single request), and the stdio backward-compatibility probe.

## Using server/discover as the stdio backward-compatibility probe

On stdio there is no per-request HTTP status code to drive fallback, so a client cannot use a status code to detect a legacy server. A client supporting both modern servers (per-request `_meta`) and legacy servers (the `initialize` handshake) SHOULD send `server/discover` first and use the outcome to decide which mode to use. This is the practical replacement for `initialize` as a first message on stdio in the 2026-07-28 revision.

## Which MCP operations must return ttlMs and cacheScope caching hints

Servers MUST include caching hints on results with `resultType: "complete"` returned by exactly six operations: `server/discover`, `tools/list`, `prompts/list`, `resources/list`, `resources/templates/list`, and `resources/read`. Interim results with `resultType: "input_required"` are not cacheable and carry no caching hints at all. Caching is complementary to change notifications; both mechanisms coexist rather than one replacing the other.

## What ttlMs means and how to handle zero, absent, or negative values

`ttlMs` is an integer number of milliseconds telling the client how long it MAY consider the result fresh, analogous to HTTP `Cache-Control: max-age`. If `ttlMs` is `0` the response SHOULD be considered immediately stale. If positive, the client SHOULD consider it fresh for that many milliseconds after receipt. If absent, clients SHOULD assume `0` and fall back to their own heuristics or notifications — which should only occur with older server versions. If negative, clients SHOULD ignore it and treat it as `0`. Servers MUST provide a value greater than or equal to zero. TTL is a freshness hint, not a guarantee: servers MAY change the underlying data before it expires.

## Computing cache freshness from ttlMs and why not to poll on the TTL

A client records the local time the response arrived, and the response is fresh while now is less than that time plus `ttlMs`; once expired the client SHOULD re-fetch on next access. Clients SHOULD NOT treat the TTL as a polling interval triggering automatic background re-fetches — it is checked when the data is needed, not on a timer — and implementations that do poll MUST apply jitter and backoff. Clients MAY re-fetch early if they believe the data changed, for example after an unexpected method-not-found or invalid-params error on a tool call, and MAY serve stale responses when errors occur during re-fetching.

## cacheScope public versus private and the risk of sharing authenticated results

`cacheScope` controls who may cache a response and takes one of two values, `"public"` or `"private"`. `"public"` means the response contains no user-specific data, so any client, shared gateway, or caching proxy MAY store and serve it to any user; it suits lists of tools, prompts, and resource templates identical for all users. `"private"` means the response contains data not meant to be shared between callers: cached responses MAY be reused within the same authorization context but MUST NOT be shared across authorization contexts, so a different access token requires a different cache. The security section warns that a `"public"` result from an authenticated endpoint may still be cached and shared outside the original authorization context, so implementors MUST apply per-primitive access controls and MUST NOT rely on `cacheScope` alone to prevent unauthorized access.

## What forms the cache key for a cached MCP response

A cached response is identified by the request method together with the request parameters that affect the result — the `uri` for `resources/read`, or the `cursor` for a paginated list. Clients MUST NOT serve a cached response for a request whose method or parameters differ from the one that produced it. Results produced by retrying a request through the multi round-trip requests mechanism — requests carrying `inputResponses` or `requestState` — MUST NOT be cached at all, because they depend on inputs that are not part of the cache key.

## How listChanged notifications interact with a still-fresh ttlMs cache

TTL caching and server-push notifications are complementary. A server MAY provide `ttlMs` without advertising `listChanged: true`, in which case the client relies entirely on TTL freshness. A server MAY advertise both, letting the client avoid unnecessary re-fetches between notifications while the notification acts as an immediate invalidation signal. When a relevant notification such as `notifications/tools/list_changed` arrives while a cached response is still fresh, the notification invalidates that response and it should be treated as immediately stale.

## Caching paginated list pages: per-page ttlMs and cacheScope consistency

When a list result is paginated, each page is an independently cacheable response. Each page carries its own `ttlMs` and its freshness clock starts when that page arrived, and servers MAY return different values on different pages — a longer TTL for stable early pages, a shorter one for the last. When a cached page expires the client SHOULD re-fetch that page using its cursor. There is no cross-page consistency guarantee, so clients may observe duplicates or gaps if data changes mid-walk; clients needing a consistent snapshot SHOULD re-fetch from the beginning without a cursor, and if a cursor becomes invalid SHOULD discard all cached pages and restart. Servers MUST apply the same `cacheScope` to every page of a given list request.

## cursor and nextCursor fields in paginated MCP list requests and responses

Pagination in MCP is opaque cursor-based rather than numbered pages. A paginated response returns the current page alongside an optional `nextCursor` when more results exist, plus `resultType: "complete"`, `ttlMs`, and `cacheScope`. To continue, the client re-issues the same list method with the token in a `cursor` parameter under `params`. Page size is determined entirely by the server and clients MUST NOT assume a fixed page size.

```json
{"jsonrpc": "2.0", "id": "124", "method": "resources/list",
 "params": {"cursor": "eyJwYWdlIjogMn0="}}
```

## Which MCP operations support cursor pagination

Exactly four operations support pagination in this revision: `resources/list`, `resources/templates/list`, `prompts/list`, and `tools/list`. Servers SHOULD provide stable cursors and handle invalid cursors gracefully. Clients SHOULD treat a missing `nextCursor` as the end of results and SHOULD support both paginated and non-paginated flows. Note that `completion/complete` is not in this list — it caps values at 100 and reports overflow via `hasMore`, not via a cursor.

## Treating pagination cursors as opaque and the invalid-cursor error code

Clients MUST treat cursors as opaque tokens: do not parse or modify them, and make no determination from the cursor value other than whether a non-null value was provided. The spec calls out the specific trap that an empty string is a valid cursor and therefore MUST NOT be treated as the end of results — only a missing `nextCursor` ends the walk. Invalid cursors SHOULD result in an error with code `-32602` (Invalid params).

## completion/complete request shape with ref, argument, and context.arguments

Clients request autocompletion for prompt arguments and resource template arguments with `completion/complete`. `params.ref` identifies what is being completed and is either a `PromptReference` or a `ResourceTemplateReference`. `params.argument` is an object with `name` and `value`, the current partial value the user typed. For prompts or URI templates with multiple arguments, clients should include already-resolved completions in `params.context.arguments`, a mapping of argument names to values, so the server can produce context-sensitive suggestions.

```json
{"method": "completion/complete",
 "params": {"ref": {"type": "ref/prompt", "name": "code_review"},
            "argument": {"name": "framework", "value": "fla"},
            "context": {"arguments": {"language": "python"}}}}
```

## CompleteResult fields values, total, and hasMore with the 100-item cap

The result has `resultType: "complete"` and a `completion` object with three fields, and the two numeric-sounding ones answer DIFFERENT questions — do not substitute one for the other.

`values` is an array of suggestion strings ranked by relevance, capped at 100 items per response. **`hasMore` is the field that says more results exist**: a boolean, and the only field a client should read to decide whether the list was truncated. `total` is something else — an OPTIONAL count of how many matches exist altogether. It is a size, not a truncation signal, it may be absent entirely, and a client that infers "there are more" from `total` alone is reading the wrong field. Servers SHOULD return suggestions sorted by relevance, implement fuzzy matching where appropriate, rate limit requests, and validate all inputs; clients SHOULD debounce rapid requests, cache results where appropriate, and handle missing or partial results gracefully.

## completions capability, ref/prompt versus ref/resource, and completion error codes

Servers supporting completions MUST declare the `completions` capability, an empty object under `capabilities`. Two reference types exist: `ref/prompt` references a prompt by name, as in `{"type": "ref/prompt", "name": "code_review"}`, and `ref/resource` references a resource URI or template, as in `{"type": "ref/resource", "uri": "file:///{path}"}` — note that `ref/prompt` uses `name` while `ref/resource` uses `uri`. Servers SHOULD return `-32601` when the method is not found because the capability is unsupported, `-32602` for an invalid prompt name or missing required arguments, and `-32603` for internal errors. Implementations MUST validate all completion inputs, rate limit, control access to sensitive suggestions, and prevent completion-based information disclosure.

## MCP logging is deprecated as of 2026-07-28 and what to migrate to

The Logging feature is deprecated as of protocol version `2026-07-28` under SEP-2577. It appears in the deprecated features registry, and under the feature lifecycle policy it remains in the specification for at least twelve months before becoming eligible for removal. New implementations SHOULD NOT adopt it. Existing implementations SHOULD migrate to logging to `stderr` for stdio transports, or to OpenTelemetry for structured observability. Servers that still emit log notifications MUST declare the `logging` capability as an empty object.

## Setting MCP log verbosity per request via io.modelcontextprotocol/logLevel

In this revision clients control logging verbosity per request through `_meta`, not through a session-wide level-setting RPC — the 2026-07-28 logging page documents no `logging/setLevel` request, and code written against earlier revisions that calls it is invoking a method this revision does not define. To receive log messages for a specific request the client includes `io.modelcontextprotocol/logLevel` in that request's `_meta`. The server MUST NOT emit `notifications/message` for a request that does not include this field. When present, the server MAY send `notifications/message` at or above the requested level on that request's response stream, before the final response. An unrecognized level SHOULD be rejected with `-32602`.

## notifications/message is request-scoped and must not appear on a listen stream

Servers send log messages with `notifications/message`, whose `params` carry `level` (the severity name), `logger` (an optional logger name such as `"database"`), and `data` (arbitrary JSON-serializable content). This notification is strictly request-scoped in 2026-07-28: the server MUST NOT deliver it on a `subscriptions/listen` stream or any stream other than the one carrying the response to the request that set `io.modelcontextprotocol/logLevel`. Servers SHOULD rate limit log messages, include relevant context, use consistent logger names, and remove sensitive information; log messages MUST NOT contain credentials or secrets, personal identifying information, or internal system details that could aid attacks.

## MCP log level names accepted in level and logLevel

The protocol follows the standard syslog severity levels from RFC 5424 section 6.2.1. The eight recognized names, least to most severe, are `debug` (detailed debugging information), `info` (general informational messages), `notice` (normal but significant events such as configuration changes), `warning` (warning conditions such as deprecated feature usage), `error` (operation failures), `critical` (system component failures), `alert` (action must be taken immediately, such as data corruption detected), and `emergency` (system is unusable). These are valid both in the `level` field of `notifications/message` and in the `io.modelcontextprotocol/logLevel` request `_meta` field.


## Delivering an elicitation/create request from a server in MCP 2026-07-28

**The field that carries the request is `inputRequests`.** A server
delivers `elicitation/create` by returning an `InputRequiredResult` and
putting the request object into its `inputRequests` map. That map is the
carrier — naming `InputRequiredResult` without naming `inputRequests`
does not say where the request actually travels.

A server no longer sends `elicitation/create` to the client as a server-initiated JSON-RPC request. Instead, while processing a client request such as `tools/call`, the server returns an `InputRequiredResult` whose `inputRequests` carries the `elicitation/create` request object (a `method` plus `params`); the client gathers the user's answer and replays the original request with the answer in `inputResponses`. Code written against earlier revisions that registers a client-side handler for an inbound `elicitation/create` request, or awaits its JSON-RPC response, no longer matches the protocol. The correlation fields of the older asynchronous design — `elicitationId` and the `notifications/elicitation/complete` notification — are gone; continuation is carried instead by the `requestState` the server echoes through MRTR.

## Declaring the elicitation client capability with form and url modes

Clients supporting elicitation MUST declare the `elicitation` capability in `_meta.io.modelcontextprotocol/clientCapabilities` on each request — not once in an `initialize` handshake as in earlier revisions. The capability object holds per-mode sub-objects `form` and `url`. For backwards compatibility an empty `elicitation` object is equivalent to declaring `form` mode only. A client declaring `elicitation` MUST support at least one mode, and servers MUST NOT send elicitation requests using modes the client has not declared.

```json
{"io.modelcontextprotocol/clientCapabilities":
  {"elicitation": {"form": {}, "url": {}}}}
```

## Required elicitation/create parameters: mode and message

Every `elicitation/create` request MUST include a `message` — a human-readable string explaining why the interaction is needed — and MAY include a `mode` whose only values are `"form"` and `"url"`. `mode` is optional for form mode and defaults to `"form"` when omitted; servers MAY omit it on form requests and clients MUST treat a request with no `mode` as form mode. The modes differ in where the data lands: `"form"` is in-band structured collection with optional schema validation and the data is exposed to the MCP client, while `"url"` is an out-of-band interaction via URL navigation where data other than the URL itself is not exposed to the client. Form mode additionally requires `requestedSchema`; URL mode additionally requires `url` and MUST specify `mode: "url"` explicitly.

## Allowed types and string formats in an elicitation requestedSchema

The `requestedSchema` of a form-mode request is a JSON Schema restricted to a deliberately small subset: flat objects with primitive properties only. The permitted property schemas are string (`minLength`, `maxLength`, `format`, `default`), number or integer (`minimum`, `maximum`, `default`), boolean (`default`), and enum forms. Every primitive carries optional `title` and `description`, and clients supporting defaults SHOULD pre-populate fields from `default`. The only supported string `format` values are `email`, `uri`, `date`, and `date-time`. The top-level object uses `properties` plus `required`. Nested structures, arrays of objects beyond the enum forms, and other advanced JSON Schema features are intentionally unsupported, so a server emitting `$ref`, nested objects, or `allOf` is outside the spec.

## Single-select and multi-select enum shapes in an elicitation requestedSchema

Enumerated choices come in four exact shapes. A single-select without display titles is `"type": "string"` with an `enum` array and an optional `default`. A single-select with titles replaces `enum` with `oneOf`, an array of `{"const": ..., "title": ...}` members, so the wire value and label can differ. Multi-select without titles is `"type": "array"` with `minItems`/`maxItems` and an `items` schema of `{"type": "string", "enum": [...]}`, with `default` as an array. Multi-select with titles keeps the array wrapper but uses `items.anyOf` holding const/title members. Note the asymmetry that trips implementers: single-select titled enums use `oneOf`, multi-select titled enums use `anyOf` nested inside `items`.

## The accept, decline and cancel action values in an elicitation response

An elicitation result uses a three-action model in the `action` field, applying to both form and URL modes. `"accept"` means the user explicitly approved and submitted; in form mode the accompanying `content` object holds the submitted data matching `requestedSchema`, and in URL mode `content` is omitted. `"decline"` means the user explicitly refused, and `content` is typically omitted. `"cancel"` means the user dismissed without an explicit choice — closed the dialog, clicked outside, pressed Escape, or the browser failed to load. Servers should treat the three distinctly: process the data on accept, offer alternatives on decline, and consider prompting again later on cancel. Collapsing decline and cancel into one "rejected" branch loses the distinction the spec draws.

## URL mode elicitation: the url parameter and what accept actually means

URL mode sends the user out of band to an external page for interactions that must not pass through the MCP client — auth flows, payment processing, credential entry. The request MUST carry `mode: "url"`, a `message`, and a `url` parameter containing a valid URL. The client's result is just `{"action": "accept"}` with no `content`, and crucially `accept` signals only that the user consented to the interaction — it does not mean the interaction completed. The outcome happens out of band and the client is never directly told. When the client retries the original request, the server uses the echoed `requestState` to decide whether the out-of-band step finished, returning either the final result or another `InputRequiredResult`. URL mode is explicitly not for authorizing the MCP client's access to the MCP server — that remains MCP authorization.

## Why form mode elicitation must never ask for passwords or API keys

The spec draws a hard trust-and-safety line between the modes. Servers MUST NOT use form mode elicitation to request sensitive information such as passwords, API keys, access tokens, or payment credentials, and MUST use URL mode for interactions involving such information. "Sensitive information" here means secrets and credentials that grant access or authorize transactions; general contact or profile data such as a name or email address is not categorically prohibited. Routing secrets through URL mode keeps them out of the LLM context, the MCP client, and intermediate MCP servers. Correspondingly, servers MUST NOT transmit credentials obtained through URL mode back to the MCP client, and MUST NOT rely on URL mode elicitation to authorize users for the server itself.

## Client rules for safely presenting an elicitation URL to the user

A client MUST NOT automatically pre-fetch the URL or its metadata, MUST NOT open it without explicit user consent, MUST show the full URL to the user for examination before consent, and MUST open it in a manner preventing the client or LLM from inspecting page content or user inputs — the spec cites iOS `SFSafariViewController` as acceptable and `WkWebView` as not. Clients SHOULD highlight the URL's domain to mitigate subdomain spoofing, SHOULD warn on ambiguous or suspicious URIs such as those containing Punycode, and SHOULD NOT render URLs as clickable in any elicitation field except the `url` field of a URL mode request. Mirroring this, servers MUST NOT put sensitive end-user information or credentials in the URL, MUST NOT supply a pre-authenticated URL to a protected resource, and SHOULD use HTTPS outside development.

## Preventing the URL mode elicitation phishing attack by verifying who opens the link

Because URL mode hands the client a link, an attacker can forward it to someone else. The spec walks the attack: malicious user Alice triggers an elicitation on a benign server, the server generates a third-party authorization URL as an OAuth client, Alice tricks victim Bob into opening it, Bob completes the authorization believing it is his own, and the server binds the resulting third-party tokens to Alice's identity — an account takeover. The server therefore MUST verify the identity of the user who opens the URL before accepting information, and MUST ensure the user who started the elicitation is the one who completes the flow. The recommended pattern is to elicit a first-party connect URL rather than the third-party authorize endpoint, and have that page compare the authoritative `sub` claim from the MCP authorization server against the browser session cookie's subject before redirecting onward. Servers MUST NOT treat client-provided user identification as authoritative, since it can be forged.

## Sampling is deprecated in MCP 2026-07-28 and what to migrate to

The Sampling feature is deprecated as of protocol version `2026-07-28` under SEP-2577. Per the feature lifecycle policy it remains in the specification for at least twelve months before becoming eligible for removal, so `sampling/createMessage` is still fully specified and existing implementations keep working. New implementations SHOULD NOT adopt sampling, and existing ones SHOULD migrate to integrating directly with LLM provider APIs rather than borrowing the client's model. The practical consequence for a server author starting today: call an LLM provider yourself with your own credentials — the "no server API keys necessary" benefit that motivated sampling is explicitly being traded away.

## Requesting a model completion with sampling/createMessage in this revision

Like elicitation, `sampling/createMessage` is no longer server-initiated: to request an LLM generation while processing a client request, the server returns an `InputRequiredResult` containing a `sampling/createMessage` request in `inputRequests`, and the client returns the generation inside `inputResponses` when it replays the request. The `params` carry `messages` (each with a `role` of `"user"` or `"assistant"` and a `content` field), plus optional `modelPreferences`, `systemPrompt`, `includeContext`, `temperature`, `stopSequences`, `metadata`, `tools`, `toolChoice`, and the required `maxTokens`. A content block is `text`, `image` (base64 `data` plus `mimeType`), or `audio`. The message list SHOULD NOT be retained between separate requests.

## Declaring the sampling capability, sampling.tools and sampling.context

Clients supporting sampling MUST declare the `sampling` capability inside `_meta.io.modelcontextprotocol/clientCapabilities` on each request — a per-request declaration, not an initialize-time capability object as in pre-2026-07-28 revisions. Basic support is the empty object. Tool-enabled sampling requires the nested `"sampling": {"tools": {}}` sub-capability: clients MUST declare `sampling.tools` to receive tool-enabled sampling requests, and servers MUST NOT send them to clients that have not. Context inclusion requires `"sampling": {"context": {}}`, itself deprecated along with the `includeContext` values it gates. Feature-detecting a client by looking for `capabilities.sampling` on the initialize response is the old shape and will not find these declarations.

## Steering model choice with modelPreferences hints and priority values

The optional `modelPreferences` object lets a server express what kind of model it wants without naming one, since the client may use a different provider entirely. It carries three normalized priority values in the range 0–1: `costPriority` (higher prefers cheaper models), `speedPriority` (higher prefers lower latency), and `intelligencePriority` (higher prefers more capable models). It also carries `hints`, an array of objects each with a `name` field such as `{"name": "claude-3-sonnet"}`. Hints are treated as substrings matching model names flexibly, multiple hints are evaluated in order of preference, clients MAY map a hint to an equivalent model from a different provider, and hints are advisory — the client makes the final selection.

## Which sampling parameters a client must respect versus may ignore

Sampling generation is tuned by four parameters and the spec is explicit about which bind the client. `maxTokens` is required and the client MUST respect it. `temperature` controls randomness, with the valid range depending on the provider. `stopSequences` is an array of sequences that stop generation. `metadata` holds provider-specific parameters. The client MAY modify or ignore `temperature`, `stopSequences`, and `metadata` — for instance because its chosen model does not support them. The optional `systemPrompt` MAY likewise be modified or ignored without telling the server, and the same latitude applies to `includeContext` where honoring it would mean sharing sensitive information.

## Reading a sampling result: role, content, model and the stopReason values

A sampling result returns `role` (generations arrive as `"assistant"`), `content`, `model`, and `stopReason`. The `content` field is polymorphic: a single content block object when the response contains only one block, or an array of blocks when there are several, such as multiple parallel tool uses or mixed content — client code that always indexes `content[0]` breaks on single-block responses and vice versa. `model` is the name of the model that actually generated the message. `stopReason` gives why sampling stopped when known, and the spec defines a non-exhaustive set — `"endTurn"`, `"stopSequence"`, `"maxTokens"`, and `"toolUse"` — while noting implementations MAY supply arbitrary values, so consumers must tolerate unknown strings.

## The includeContext parameter values and their deprecation

`includeContext` tells the client what context to include in its sampling response and takes one of three values: `"none"`, `"thisServer"` (context from the requesting server), and `"allServers"` (context from all connected MCP servers). As of 2026-07-28 the values `"thisServer"` and `"allServers"` are themselves deprecated under SEP-2596 and will be removed no later than the Sampling feature as a whole. Servers SHOULD avoid them — the guidance is to omit `includeContext`, since it defaults to `"none"` — and SHOULD NOT use them unless the client has declared the `sampling.context` capability. Older MCP examples routinely set `includeContext: "thisServer"`; copying that today emits a deprecated value.

## Letting the sampled model call tools with the tools array and toolChoice modes

A sampling request can include a `tools` array so the client's LLM can call tools during generation. Each entry has `name`, `description`, and `inputSchema`, and these definitions are scoped to the sampling request — they need not correspond to registered MCP tools. The optional `toolChoice` object controls tool use through its `mode` field: `{"mode": "auto"}` lets the model decide and is the default, `{"mode": "required"}` means the model MUST use at least one tool before completing, and `{"mode": "none"}` means it MUST NOT use any. The model may emit several `tool_use` blocks in parallel. The server drives the loop: it executes the tool uses, sends a new sampling request with results appended, and repeats, optionally capping iterations and passing `toolChoice: {mode: "none"}` on the last one to force a final answer. Both parties SHOULD implement iteration limits.

## Pairing tool_use and tool_result blocks correctly in sampling messages

Two structural rules govern tool results in a sampling conversation. First, when a `"user"` message contains tool results it MUST contain only tool results — mixing `tool_result` blocks with `text`, `image`, or `audio` blocks in the same message is not allowed, a constraint that exists so the conversation maps onto provider APIs with dedicated tool roles. Second, every assistant message containing `ToolUseContent` blocks MUST be followed by a user message consisting entirely of `ToolResultContent` blocks, each tool use's `id` matched by a result carrying the same value in `toolUseId`, before any other message. A tool use left unresolved makes the sequence invalid. A `tool_result` block carries `toolUseId` and a `content` array; a `tool_use` block carries `id`, `name`, and `input`.

## Roots is deprecated in MCP 2026-07-28 and what replaces it

The Roots feature is deprecated as of protocol version `2026-07-28` under SEP-2577. Under the feature lifecycle policy it stays in the specification for at least twelve months before becoming eligible for removal, and it appears in the deprecated features registry. New implementations SHOULD NOT adopt roots. Existing implementations SHOULD migrate to passing directories or files via tool parameters, resource URIs, or server configuration — make the working directory an explicit argument of the tool that needs it, rather than asking the client for a workspace list at runtime.

## Retrieving the client's filesystem roots with a roots/list request

Roots let a client tell servers which directories and files it considers relevant. They are informational guidance, not an access-control mechanism: the protocol does not enforce that servers stay within roots. In this revision a server obtains them not by sending `roots/list` as a server-initiated request but by returning an `InputRequiredResult` containing a `roots/list` request in `inputRequests`; the client's result — returned in `inputResponses` on the retried call — is an object with a `roots` array. Each Root has `uri`, which MUST be a `file://` URI in the current specification, and an optional human-readable `name`. Clients supporting roots MUST declare the `roots` capability on each request. Clients MUST only expose roots with appropriate permissions, validate all root URIs to prevent path traversal, and implement access controls; servers SHOULD respect root boundaries and validate paths against provided roots.

## Asking for progress updates by putting a progressToken in request _meta

Progress tracking is opt-in from the requesting side. When a client wants progress updates for a request it includes a `progressToken` in that request's `_meta` object — not in `params` directly and not as a top-level JSON-RPC field. Progress tokens MUST be a string or an integer. The client can choose the token by any means, but it MUST be unique across all active requests, since that token is the only thing correlating later notifications back to the originating request. A server receiving a progress token MAY send no progress notifications at all, send them at whatever frequency it deems appropriate, and omit the total if unknown, so a client must never depend on progress arriving.

```json
{"jsonrpc": "2.0", "id": 1, "method": "some_method",
 "params": {"_meta": {"progressToken": "abc123"}}}
```

## Sending notifications/progress with progress, total and message fields

A server reports status on a long-running operation with `notifications/progress`, whose `params` carry the original `progressToken`, the current `progress` value, an optional `total`, and an optional human-readable `message` such as "Reticulating splines...". The `progress` value MUST increase with each notification even when the total is unknown, and both `progress` and `total` MAY be floating point. Progress notifications MUST only reference tokens provided in an active request associated with an in-progress operation, and MUST stop after completion. Both parties SHOULD track active progress tokens and SHOULD implement rate limiting to prevent flooding.

## Cancelling an in-progress MCP request with notifications/cancelled

A client SHOULD send `notifications/cancelled` to indicate that a request it previously issued should be terminated. The `params` carry `requestId`, the ID of the request to cancel, and an optional `reason` string. Cancellation notifications MUST only reference requests previously issued by the client and believed still in-progress. A server receiving one SHOULD stop processing, free associated resources, and not send a response for it; servers MAY ignore the notification if the request is unknown, already completed, or cannot be cancelled. Because network latency means a cancellation can arrive after a response was sent, both parties MUST handle the race gracefully and the client SHOULD ignore any response to a cancelled request that arrives afterward. Invalid cancellation notifications SHOULD simply be ignored.

## Transport-specific cancellation: closing the SSE stream versus notifications/cancelled

How a client signals cancellation depends on the transport, which is a meaningful change from treating `notifications/cancelled` as universal. On Streamable HTTP, closing the SSE response stream is the cancellation signal: the server MUST treat a client disconnect as cancellation of that request, and no `notifications/cancelled` message is required or expected. On stdio there is no per-request stream to close, so the client MUST send `notifications/cancelled` referencing the request ID. In the server-to-client direction the notification is tightly restricted: a server MUST send `notifications/cancelled` referencing a `subscriptions/listen` request ID when it tears down that subscription stream, and MUST NOT send it for any other purpose.

## Request timeouts and whether progress notifications reset the clock

Implementations SHOULD establish timeouts for all sent requests to prevent hung connections and resource exhaustion, and when no response arrives within the timeout the sender SHOULD cancel the request and stop waiting. Cancelling on timeout means what transport-specific cancellation means: closing the response stream on Streamable HTTP, sending `notifications/cancelled` on stdio. SDKs and middleware SHOULD allow these timeouts to be configured per request. Implementations MAY reset the timeout clock on receiving a `notifications/progress` for the request, since that implies work is happening, but SHOULD always enforce a maximum timeout regardless, so a misbehaving peer cannot hold a request open indefinitely by emitting progress forever.


## Getting an OAuth client_id when the MCP client and authorization server have never met

Client ID Metadata Documents (CIMD) are new in MCP revision 2026-07-28 and are the intended default for the common case where client and server have no pre-existing relationship. The mechanism is specified in `draft-ietf-oauth-client-id-metadata-document-00` and lets a client use an HTTPS URL as its `client_id` directly, with the URL pointing at a JSON document containing the client metadata. Authorization servers and MCP clients SHOULD support it. When the authorization server sees a URL-formatted `client_id`, it fetches that URL, validates the document, and uses `client_name` on the consent screen. Models trained on earlier MCP revisions reach for Dynamic Client Registration here; in 2026-07-28 that path is deprecated and CIMD is the replacement.

## Hosting a client ID metadata document at an HTTPS URL

An MCP client using CIMD MUST host its metadata document at an HTTPS URL, and the `client_id` URL MUST use the `https` scheme and contain a path component — for example `https://example.com/client.json`. The document MUST include at least `client_id`, `client_name`, and `redirect_uris`. Clients MUST ensure the `client_id` value inside the metadata matches the document URL exactly, so the URL is self-describing and cannot be pointed at someone else's identity. Clients MAY use `private_key_jwt` for client authentication with appropriate JWKS configuration. Other fields shown in the spec's example include `client_uri`, `logo_uri`, `grant_types`, `response_types`, and `token_endpoint_auth_method`.

```json
{"client_id": "https://app.example.com/oauth/client-metadata.json",
 "client_name": "Example MCP Client",
 "redirect_uris": ["http://127.0.0.1:3000/callback"],
 "grant_types": ["authorization_code"], "response_types": ["code"],
 "token_endpoint_auth_method": "none"}
```

## Validating a URL-formatted client_id at the authorization server

An authorization server implementing CIMD SHOULD fetch metadata documents when it encounters URL-formatted `client_id` values. It MUST validate that the fetched document's `client_id` matches the URL exactly, MUST validate that the structure is valid JSON containing the required fields, and MUST validate redirect URIs presented in an authorization request against those in the document's `redirect_uris`. It SHOULD cache metadata respecting HTTP cache headers rather than re-fetching per authorization. On validation failure the server returns `error=invalid_client` or `error=invalid_request`; on success it displays a consent page carrying the document's `client_name` and the redirect URI hostname.

## Checking client_id_metadata_document_supported before using CIMD

Authorization servers advertise CIMD support by including `"client_id_metadata_document_supported": true` in their OAuth Authorization Server Metadata document (the RFC 8414 metadata retrieved during discovery). MCP clients SHOULD check for this capability before attempting to use an HTTPS URL as `client_id`, and MAY fall back to Dynamic Client Registration or pre-registration if unavailable. This field is the CIMD analogue of `registration_endpoint`, the RFC 7591 field a client checks to see whether the deprecated DCR path is available.

## Choosing between pre-registration, Client ID Metadata Documents, and Dynamic Client Registration

MCP defines exactly three client registration mechanisms and a client MUST obtain a client ID through one of them before initiating the authorization flow: Client ID Metadata Documents (when client and server have no prior relationship — the most common case), pre-registration (when they have an existing relationship), and Dynamic Client Registration (for backwards compatibility). Clients supporting all options SHOULD use this priority order: first use pre-registered client information if available; second use CIMD if the server advertises `client_id_metadata_document_supported`; third use DCR as a fallback if the server advertises a `registration_endpoint`; fourth prompt the user to enter client information. Note that pre-registration ranks above CIMD, which reverses the naive assumption that the newest mechanism is always tried first.

## Dynamic Client Registration is deprecated in MCP 2026-07-28

The spec carries an explicit warning: Dynamic Client Registration is deprecated, new implementations should use Client ID Metadata Documents instead, and the option remains available only for backwards compatibility with authorization servers that do not support CIMD. Clients and authorization servers MAY support RFC 7591 DCR to obtain client IDs without user interaction. This is a downgrade from earlier MCP revisions, where RFC 7591 was the recommended path for the no-prior-relationship case — guidance written against 2025-03-26 or 2025-06-18 that treats `POST /register` as the primary way to get a `client_id` is now writing the deprecated protocol. Note the keyword strength: DCR is MAY, while CIMD is SHOULD.

## Setting application_type to native or web during Dynamic Client Registration

When authorization servers support OpenID Connect and Dynamic Client Registration, they may enforce additional redirect-URI constraints based on the `application_type` parameter from OpenID Connect Dynamic Client Registration 1.0. MCP clients MUST specify an appropriate `application_type` during DCR. Omitting it defaults to `"web"` under OIDC, which conflicts with native-style redirect URIs such as `http://localhost:3000/callback`; non-OIDC servers safely ignore the parameter. Native applications — desktop apps, mobile apps, CLI tools, and locally-hosted web apps accessed via localhost — SHOULD use `"native"`, while web applications served from a non-local host SHOULD use `"web"`. Clients MUST be prepared to handle registration failures due to redirect URI constraints, SHOULD surface a meaningful error, and MAY retry with an adjusted `application_type`.

## Keying stored client credentials to the authorization server's issuer

Clients using pre-registered credentials, or persisting credentials obtained via DCR, MUST associate those credentials with the specific authorization server that issued them, keyed by the server's `issuer` identifier. When the authorization server changes — detected via updated protected resource metadata — clients MUST NOT reuse credentials from a different authorization server and MUST re-register with the new one. If the server indicated by protected resource metadata no longer matches the one the credentials were registered with, clients SHOULD surface an error rather than silently attempting to use mismatched credentials. Client IDs based on Client ID Metadata Documents are exempt: they are portable across authorization servers because they are self-hosted HTTPS URLs resolved on demand.

## Returning 401 with a WWW-Authenticate header pointing at resource_metadata

MCP servers MUST implement one of two discovery mechanisms so clients can locate the authorization server, and the first is the `WWW-Authenticate` header: include the resource metadata URL under the `resource_metadata` parameter when returning `401 Unauthorized`, per RFC 9728 Section 5.1. MCP clients MUST be able to parse `WWW-Authenticate` headers and respond appropriately to 401 responses, MUST use the resource metadata URL from the parsed header when present, and only fall back to well-known URI probing when it is absent. Servers SHOULD also include a `scope` parameter per RFC 6750 Section 3 to tell the client which scopes the resource requires.

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource", scope="files:read"
```

## Falling back to the /.well-known/oauth-protected-resource URIs

The second permitted discovery mechanism is serving protected resource metadata at a well-known URI per RFC 9728, in one of two locations: at the path of the server's MCP endpoint, so a server at `https://example.com/public/mcp` hosts metadata at `https://example.com/.well-known/oauth-protected-resource/public/mcp`; or at the root, `https://example.com/.well-known/oauth-protected-resource`. MCP clients MUST support both discovery mechanisms, and when no `resource_metadata` parameter is present in a `WWW-Authenticate` header they MUST fall back to constructing and requesting the well-known URIs in that order — the path-inserted sub-path form first, then the root form. If neither is found, the client aborts or uses pre-configured values.

## Which .well-known URL to probe for authorization server metadata

MCP uses the default `oauth-authorization-server` well-known URI suffix from RFC 8414 Section 3.1 and defines no application-specific suffix. Because issuer URL formats vary, MCP clients MUST attempt multiple well-known endpoints in a fixed priority order depending on whether the issuer URL has a path component. For an issuer with a path such as `https://auth.example.com/tenant1`, clients MUST try in order: `https://auth.example.com/.well-known/oauth-authorization-server/tenant1` (RFC 8414 path insertion), then `https://auth.example.com/.well-known/openid-configuration/tenant1` (OIDC path insertion), then `https://auth.example.com/tenant1/.well-known/openid-configuration` (OIDC path appending). For an issuer without a path, clients MUST try `/.well-known/oauth-authorization-server` then `/.well-known/openid-configuration`. Authorization servers MUST provide at least one of RFC 8414 or OpenID Connect Discovery 1.0; clients MUST support both.

## Rejecting authorization server metadata whose issuer does not match

After retrieving an authorization server metadata document, MCP clients MUST validate it per RFC 8414 Section 3.3 or OpenID Connect Discovery Section 4.3: the `issuer` value in the document MUST be identical to the issuer identifier used to construct the well-known URL. If they differ, the client MUST NOT use the metadata. The spec gives the concrete attack: a document fetched from `https://attacker.example/.well-known/oauth-authorization-server` containing `"issuer": "https://honest.example"` MUST be rejected. Without this check, an attacker-controlled authorization server listed in or injected into protected resource metadata could claim an honest server's identity and collect authorization codes intended for it. This validated `issuer` is what the client later compares against the `iss` parameter in the authorization response.

## Reading authorization_servers from protected resource metadata and handling multiple servers

MCP servers MUST implement OAuth 2.0 Protected Resource Metadata (RFC 9728) to indicate authorization server locations, and the document MUST include the `authorization_servers` field containing at least one authorization server. MCP clients MUST use Protected Resource Metadata for authorization server discovery. A document may define multiple authorization servers, and responsibility for selecting one lies with the client, following RFC 9728 Section 7.6. Each listed server is independent, and consistent with RFC 6749 Section 2.2 client identifiers are unique to the authorization server that issued them. Clients MUST maintain separate registration state — credentials and tokens — per authorization server, and MUST NOT assume credentials valid for one will be accepted by another.

## Validating the iss parameter on the authorization response per RFC 9207

Before redirecting the user-agent, the client MUST record the `issuer` value from the selected server's validated metadata and associate it with the same per-request record holding the PKCE code verifier and `state`. MCP authorization servers SHOULD include the `iss` parameter in authorization responses, including error responses, per RFC 9207 Section 2, and servers that include it MUST advertise this by setting `authorization_response_iss_parameter_supported` to true. On receiving the response, clients MUST apply RFC 9207 Section 2.4 validation before transmitting the authorization code to any token endpoint: if the metadata flag is true and `iss` is present, compare it to the recorded issuer by simple string comparison; if the flag is true and `iss` is absent, reject; if the flag is false or absent but `iss` is present, still compare it; if both are absent, proceed. This defends against mix-up attacks where an attacker controlling one authorization server tries to make the client send it a code issued by an honest one.

## Comparing the iss value byte-for-byte without URL normalization

After decoding the `iss` value from the form-urlencoded authorization response, clients MUST NOT apply scheme or host case folding, default-port elision, trailing-slash, or percent-encoding normalization before comparison with the recorded issuer. Only simple string comparison per RFC 3986 Section 6.2.1 is permitted, because any normalization widens the set of strings that match the honest issuer and gives an attacker room to construct a near-miss issuer identifier. The validation provides no protection at all if the expected issuer came from an unvalidated source rather than a metadata document validated during discovery. This applies equally to error responses: on mismatch the client MUST NOT act on or display `error`, `error_description`, or `error_uri`.

## Sending the resource parameter with the MCP server's canonical URI per RFC 8707

MCP clients MUST implement Resource Indicators for OAuth 2.0 (RFC 8707) to specify the target resource for which the token is requested. The `resource` parameter MUST be included in both authorization requests and token requests, MUST identify the MCP server the client intends to use the token with, and MUST use the canonical URI of that server, aligning with the `resource` field in RFC 9728 protected resource metadata. Clients MUST send it regardless of whether the authorization server supports it. Valid canonical URIs include `https://mcp.example.com/mcp`, `https://mcp.example.com`, and `https://mcp.example.com:8443`; invalid examples are `mcp.example.com` (missing scheme) and `https://mcp.example.com#fragment` (contains a fragment). Clients SHOULD provide the most specific URI they can and SHOULD use the form without a trailing slash.

## Sending the access token in the Authorization Bearer header on every request

Access token handling MUST conform to OAuth 2.1 Section 5. The MCP client MUST use the `Authorization` request header field in the form `Authorization: Bearer <access-token>`, and authorization MUST be included in every HTTP request from client to server — not only the first, and not established once per session. Access tokens MUST NOT be included in the URI query string, since query strings leak into logs, referrer headers, and browser history. MCP clients MUST NOT send tokens to the MCP server other than ones issued by the MCP server's own authorization server.

## Validating that an access token was issued for this MCP server, and refusing token passthrough

MCP servers, acting as OAuth 2.1 resource servers, MUST validate access tokens per OAuth 2.1 Section 5.2, and MUST validate that tokens were issued specifically for them as the intended audience per RFC 8707 Section 2. Servers MUST only accept tokens intended for themselves, MUST reject tokens that do not include them in the audience claim, and MUST NOT accept or transit any other tokens. Invalid or expired tokens MUST receive an HTTP 401. Validation must happen before processing the request. If the MCP server makes requests to upstream APIs it may act as an OAuth client to them, but the token used upstream is a separate token issued by the upstream authorization server — the MCP server MUST NOT pass through the token it received from the MCP client.

## Responding with 403 insufficient_scope when a valid token lacks a permission

When a client makes a request with a token whose scope is insufficient during runtime operations, the server SHOULD respond with `403 Forbidden` per RFC 6750 Section 3.1 and a `WWW-Authenticate` header carrying `error="insufficient_scope"`, a `scope` parameter listing the minimum scopes needed, the `resource_metadata` URI for consistency with 401 responses, and optionally a human-readable `error_description`. This is distinct from the 401 case — and the distinction below is about **an MCP endpoint's OAuth flow specifically**, not about HTTP in general. At an MCP endpoint: 401 means the access token is missing, expired or invalid, 403 means the token is valid but its scopes are insufficient, and 400 means a malformed authorization request. In plain HTTP the ordinary split still holds and nothing here changes it: **401 Unauthorized is about authentication** — the caller has not proved who they are — **and 403 Forbidden is about authorization** — identity is established and the caller still may not do this. Do not answer a general HTTP question with the MCP scope wording. Servers SHOULD include all scopes required for the current operation in a single challenge — challenging incrementally forces multiple authorization round-trips — SHOULD be consistent in their strategy, and MUST account for scope hierarchies where a broader scope implies narrower ones.

## Step-up authorization: unioning previously requested scopes with the challenge scopes

Clients receive scope errors either during initial authorization or at runtime as `insufficient_scope`, and SHOULD respond by requesting a new access token with an increased scope set via a step-up flow. Clients acting on behalf of a user SHOULD attempt step-up; clients acting on their own behalf MAY attempt it or abort immediately. The flow is: parse the error from the response or `WWW-Authenticate` header; determine required scopes by computing the union of the client's previously requested scope set and the scopes from the current challenge — the critical step, because servers emit per-operation scope challenges and are not required to echo previously granted scopes, so a client requesting only the challenged scope silently loses permissions it already had; initiate re-authorization with the determined set; and retry the original request no more than a few times before treating it as a permanent failure. Scope accumulation is explicitly a client-side responsibility.

## Choosing which scopes to request in the initial authorization handshake

MCP servers SHOULD include a `scope` parameter in the `WWW-Authenticate` header per RFC 6750 Section 3 to indicate the scopes required, giving clients immediate guidance and preventing them from requesting excessive permissions. The scopes in the challenge MAY match `scopes_supported` from Protected Resource Metadata, be a subset or superset, or be an alternative collection that is neither — clients MUST NOT assume any set relationship and MUST treat the challenge scopes as authoritative for the current operation. For the initial handshake, clients SHOULD first use the `scope` parameter from the initial 401 `WWW-Authenticate` header if provided; second, if unavailable, use all scopes in `scopes_supported`, omitting the `scope` parameter entirely if `scopes_supported` is undefined. `scopes_supported` represents the minimal set needed for basic functionality, with additional scopes requested incrementally through step-up.

## Requesting refresh tokens and the offline_access scope

MCP clients wanting refresh tokens MUST keep them confidential in transit and storage per OAuth 2.1 Section 4.3, SHOULD include `refresh_token` in their `grant_types` client metadata, and MAY add `offline_access` to the `scope` parameter when the authorization server metadata lists it in `scopes_supported`. Clients MUST NOT assume refresh tokens will be issued — the authorization server retains discretion. MCP servers acting as protected resources SHOULD NOT include `offline_access` in the `WWW-Authenticate` scope challenge or in `scopes_supported`, because refresh tokens are a client-lifecycle concern, not a resource requirement. For public clients, authorization servers MUST rotate refresh tokens per OAuth 2.1 Section 4.3.1, and SHOULD issue short-lived access tokens.

## Verifying PKCE support via code_challenge_methods_supported before starting authorization

MCP clients MUST implement PKCE per OAuth 2.1 Section 7.5.2 and MUST verify PKCE support before proceeding; PKCE prevents authorization code interception and injection by requiring a secret verifier-challenge pair so only the original requestor can redeem a code. Clients MUST use the `S256` code challenge method when technically capable. Because neither OAuth 2.1 nor PKCE defines a discovery mechanism for PKCE support, clients MUST rely on authorization server metadata: with OAuth 2.0 Authorization Server Metadata, if `code_challenge_methods_supported` is absent the server does not support PKCE and clients MUST refuse to proceed. With OpenID Connect Discovery 1.0 the field is not formally defined but is commonly included, and MCP clients MUST verify its presence and MUST refuse to proceed if absent; authorization servers providing OIDC Discovery MUST include it to ensure MCP compatibility.

## Preventing open redirection with registered redirect URIs, state, and HTTPS

An attacker may craft malicious redirect URIs to direct users to phishing sites. MCP clients MUST have redirect URIs registered with the authorization server, and authorization servers MUST validate exact redirect URIs against pre-registered values — exact matching, not prefix or wildcard. MCP clients SHOULD use and verify `state` parameters in the authorization code flow and discard results that do not include, or that mismatch, the original `state`. Authorization servers MUST take precautions to prevent redirecting user agents to untrusted URIs per OAuth 2.1 Section 7.12.2, and SHOULD only automatically redirect if they trust the redirection URI. Under Communication Security, all authorization server endpoints MUST be served over HTTPS, and all redirect URIs MUST be either localhost or use HTTPS.

## SSRF and localhost impersonation risks when an authorization server fetches CIMD

Because CIMD makes the authorization server fetch an attacker-influenceable URL, servers MUST consider the security implications in Section 6 of the Client ID Metadata Document draft, and SHOULD consider Server-Side Request Forgery risks since the `client_id` URL is supplied by whoever initiates the authorization request. Client ID Metadata Documents cannot prevent localhost URL impersonation by themselves — any party can host a document claiming `http://localhost:3000/callback` as a redirect URI, so a malicious local application can impersonate a legitimate local client. Authorization servers SHOULD display additional warnings for localhost-only redirect URIs, MAY require additional attestation mechanisms, and MUST clearly display the redirect URI hostname during authorization. Servers MAY implement domain-based trust policies for accepting Client ID Metadata Documents.

## Obtaining user consent in an MCP proxy server that uses a static client ID

Attackers can exploit MCP servers acting as intermediaries to third-party APIs, producing a confused deputy vulnerability: using stolen authorization codes they can obtain access tokens without user consent. The specific hazard is an MCP proxy server presenting a single static client ID to a third-party authorization server, because that server may remember a prior consent for the static client ID and skip the consent screen for a request a different, attacker-controlled downstream client initiated. MCP proxy servers using static client IDs therefore MUST obtain user consent for each dynamically registered client before forwarding to third-party authorization servers. This sits alongside the access token privilege restriction rules: an attacker can compromise an MCP server that accepts tokens issued for other resources.


## stdio message framing: newline-delimited JSON and what may be written to stdout

The MCP stdio transport frames messages as newline-delimited JSON-RPC: the server reads messages from `stdin` and writes messages to `stdout`, one message per line, each a single JSON-RPC request, notification, or response. Messages are delimited by newlines and MUST NOT contain embedded newlines, so a serializer that pretty-prints JSON across multiple lines corrupts the stream. There is no length-prefix, no `Content-Length` header, and no framing envelope. The purity rules run both ways: the server MUST NOT write anything to `stdout` that is not a valid MCP message, and the client MUST NOT write anything to the server's `stdin` that is not a valid MCP message. This is why a stray `print()` or a startup banner breaks the connection — anything a server wants to say to a human belongs on `stderr`.

## What stderr is for on a stdio MCP server and why it is not an error signal

The server MAY write UTF-8 strings to `stderr` for any logging purpose, explicitly including informational, debug, and error messages — `stderr` is the free-form diagnostic channel alongside the strict JSON-RPC channel on `stdout`. On the client side the rules are permissive and deliberately non-inferential: the client MAY capture, forward, or ignore the server's `stderr` output, and SHOULD NOT assume that `stderr` output indicates an error condition. A client that surfaces every `stderr` line as a failure, or treats `stderr` activity as a health signal, is reading meaning into a channel the specification says carries none. This channel is also the recommended replacement for MCP's deprecated logging feature on stdio deployments.

## Reading MCP server messages from stdout: one shared channel, no per-request streams

The client reads server messages from `stdout`, one message per line, and all messages share this single channel — the stdio binding has no per-request streams, unlike Streamable HTTP where a reply can be a request-scoped SSE stream. Three kinds of message arrive interleaved: responses to client requests, correlated by JSON-RPC `id`; notifications relating to an in-flight request, such as `notifications/progress` and `notifications/message`; and notifications delivered for an active `subscriptions/listen` request. For that third kind, clients MUST correlate using the `io.modelcontextprotocol/subscriptionId` field in `_meta` — the JSON-RPC `id` cannot do this job because notifications have no `id`. Any stdio client assuming messages arrive in request order will mis-route subscription traffic.

## Why an MCP server never sends a JSON-RPC request, and returns InputRequiredResult instead

In this revision servers MUST NOT initiate JSON-RPC requests and clients do not send JSON-RPC responses; on stdio this means the server MUST NOT write JSON-RPC requests to `stdout`, and the client MUST NOT write JSON-RPC responses to the server's `stdin`. Every interaction begins with the client. When a server needs client input — sampling, elicitation, or roots — it answers the in-flight request with an `InputRequiredResult` carrying `inputRequests`, and the client retries the request as a new request, with a new JSON-RPC `id`, carrying the original params plus the matching `inputResponses`. This is a hard break from earlier MCP revisions, which allowed servers to initiate requests such as `sampling/createMessage` back at the client; code written against those revisions expects a bidirectional peer relationship that no longer exists.

## There is no initialize handshake on stdio, and a server process is not a session

The stdio binding has no `initialize` request, no `notifications/initialized`, and no connection-scoped session. MCP is stateless: every request is self-contained and carries its own protocol version and capabilities, so a client may send `tools/call` as the very first line it writes to a freshly launched server's `stdin`. Nothing in the stdio binding depends on the standard streams except the process lifecycle — the subprocess is a delivery mechanism, not a session identity. A server may not accumulate per-connection state and expect a client to have "initialized" it, and a client may not treat process start as a handshake point. Earlier revisions established exactly that connection-scoped session; libraries built before this revision emit an `initialize` call a modern stdio server has no reason to answer.

## Where the protocol version and client capabilities travel on stdio

All request metadata for stdio is carried inline in the JSON-RPC message body — there is no header layer. The protocol version, per-request capabilities, and optional client identity live under `_meta.io.modelcontextprotocol/*`, while the method name and arguments live where JSON-RPC puts them. Client capabilities in particular travel as `_meta.io.modelcontextprotocol/clientCapabilities` on every request. This is the contrast with Streamable HTTP, which mirrors selected body fields into HTTP headers so intermediaries can route without parsing the body; on stdio there is nothing to mirror into. Because the metadata is per-request, a stdio client repeats the version and capabilities on every single message rather than declaring them once.

## Cancelling an in-flight stdio request with notifications/cancelled

To cancel an in-flight request over stdio, the client MUST send a `notifications/cancelled` notification referencing the request's ID. The reason is direct: stdio is a single shared bidirectional channel, so there is no per-request stream to close. This makes `notifications/cancelled` specific to the stdio binding in this revision — on Streamable HTTP the client abandons a request by closing that request's response stream instead. On receiving the notification, servers SHOULD stop work as soon as practical and MUST NOT send any further messages for it, including progress notifications as well as the eventual response.

## Shutting down a stdio MCP server process cleanly

The client SHOULD initiate shutdown in three ordered steps: close the input stream to the child process, wait for the server to exit, and — if the server does not exit within a reasonable time — forcibly terminate the process using the mechanism appropriate for the operating system. On POSIX systems forced termination typically escalates from `SIGTERM` to `SIGKILL`; on Windows, where POSIX signals are unavailable, clients can use `TerminateProcess` or Job Objects. From the other side, servers SHOULD exit promptly when their standard input is closed or reads return end-of-file: this is the primary graceful-shutdown signal and the only portable one. The server MAY also initiate shutdown itself, by closing its output stream and exiting.

## What to do when a stdio MCP server process exits unexpectedly

If the server process exits unexpectedly, the client SHOULD restart it. Recovery is cheap precisely because of this revision's statelessness: any in-flight requests are simply lost and the client can retry them against the fresh process — there is no session to re-establish, no `initialize` to replay, and no negotiated state to reconstruct. The one thing that does not survive a restart is streaming: active `subscriptions/listen` streams must be re-established against the new process. This is a behavioral delta against session-based MCP revisions, where a dropped subprocess meant a dropped session requiring a full handshake before work could resume.

## Probing an unknown MCP server with server/discover before sending any other request

A client supporting both modern MCP versions and a legacy version requiring `initialize` SHOULD probe with `server/discover` first, setting its preferred modern version in `_meta`. The probe has exactly three outcomes. A `DiscoverResult` means modern: select a mutually supported version from `supportedVersions` and continue. A recognized modern JSON-RPC error such as `UnsupportedProtocolVersionError` means modern but not supporting the requested version — use one of the advertised versions, and do not fall back to `initialize`. Any other error, or no response within a reasonable timeout, means legacy: fall back to the `initialize` handshake. The fallback MUST NOT be keyed to one specific error code, because legacy servers answer unknown pre-`initialize` requests with implementation-defined errors — commonly `-32601` or `-32602` — or with nothing at all. Probing is RECOMMENDED even for modern-only clients, because some legacy servers do not validate that a request arrives after `initialize` and would process an era-ambiguous method such as `tools/call` under legacy semantics, so probing converts a silent misinterpretation into a deterministic failure.

## Which transports the 2026-07-28 MCP revision defines

This revision specifies two standard transport bindings: stdio, newline-delimited messages over the standard streams of a client-launched subprocess; and Streamable HTTP, each message an HTTP POST to a single MCP endpoint with replies arriving as a JSON object or a request-scoped SSE stream. Protocol semantics are identical on every transport — a transport is a binding defining how messages are framed and delivered, how request metadata is carried, and how cancellation and termination are signaled, and it does not define what messages mean. JSON-RPC messages MUST be UTF-8 encoded on any transport. A binding MUST deliver client-sent requests and notifications to the server and server-sent responses and notifications to the client; no other message direction exists. The older HTTP+SSE transport from 2024-11-05 is not one of the two standard bindings — it is a deprecated feature.

## Building a custom MCP transport over TCP or a Unix domain socket

Clients and servers MAY implement additional custom transport mechanisms, since the protocol is transport-agnostic and can run over any channel supporting bidirectional message exchange. Implementers supporting custom transports MUST preserve the JSON-RPC message format, the core message patterns, and the per-request metadata model — a custom transport may change framing and delivery, never semantics. Custom transports SHOULD document their connection establishment, message framing, and cancellation patterns. Crucially, a custom transport over a reliable bidirectional byte stream such as a Unix domain socket or TCP connection SHOULD reuse the stdio framing rather than inventing a new one: the stdio binding is just newline-delimited JSON-RPC over a byte stream, and only its subprocess-specific aspects need channel-specific equivalents.

## What a host, a client, and a server each do in the MCP architecture

MCP follows a client-host-server architecture where each host can run multiple client instances. The host process is container and coordinator: it creates and manages client instances, controls connection permissions and lifecycle, enforces security policies and consent requirements, handles user authorization decisions, coordinates AI/LLM integration, and manages context aggregation across clients. Each client is created by the host and communicates with exactly one server — a strict 1:1 relationship — attaching the protocol version and capabilities to every request, routing messages bidirectionally, managing subscriptions, and maintaining security boundaries between servers. Servers expose resources, tools, and prompts, operate independently with focused responsibilities, request client input via `InputRequiredResult` within a reply, and may be local processes or remote services. A core design principle is that servers should not read the whole conversation nor see into other servers: full conversation history stays with the host.

## MCP is a stateless protocol: every request is self-contained

The architecture page states the model directly — MCP is a stateless protocol in which every request is self-contained and carries its own protocol version and capabilities. Nothing is scoped to a connection: the client attaches version and capabilities to every request rather than declaring them once at setup. This is the single change from which most other 2026-07-28 deltas follow: it is why there is no `initialize` handshake, why a stdio subprocess is not a session, why in-flight requests can be retried against a restarted server, and why capabilities are evaluated per request rather than fixed for the lifetime of a link. Even stream state is scoped to a request rather than a connection: if the channel for a `subscriptions/listen` stream is lost, the client re-issues the request.

## How MCP capabilities are negotiated when there is no initialize step

MCP uses a capability-based negotiation system in which clients and servers declare supported features on each request. Clients include theirs in `_meta.io.modelcontextprotocol/clientCapabilities` on every request. Servers advertise theirs in the response to `server/discover`, which clients may call before any other request — discovery is optional, not a handshake. Servers declare capabilities like tool support, resource subscriptions, and prompt templates; clients declare capabilities like sampling support and elicitation handling; both parties must respect declared capabilities throughout. Each capability unlocks specific protocol features per request: implemented server features must be advertised, tool invocation requires the server to declare tool capabilities, and receiving resource update notifications requires opening a `subscriptions/listen` stream. The delta versus earlier revisions is that capabilities are no longer exchanged once during `initialize` and then assumed for the session.

## How the MCP deprecation registry works and what has been removed so far

The deprecated page is the registry of features currently in the Deprecated state under the feature lifecycle and deprecation policy defined by SEP-2596. A Deprecated feature remains part of the specification but is scheduled for removal: new implementations SHOULD NOT adopt it, and existing implementations SHOULD migrate before the earliest removal. The "earliest removal" date marks when a feature becomes eligible for removal — actual removal is a Core Maintainer decision taken during release preparation and may happen later. For features deprecated in 2026-07-28, earliest removal is the first revision released on or after 2027-07-28, i.e. one year of guaranteed life. The registry is a derived view kept consistent with per-feature deprecation notices and changelog entries, which are the normative records. The page also carries a Removed section, and as of this revision it is empty: no features have been removed under this policy yet.

## The two MCP deprecations that predate the lifecycle policy

Two registry entries were already described as deprecated before the feature lifecycle policy existed, and SEP-2596 reclassifies them as formally Deprecated under its transition provisions. The first is the Sampling field values `includeContext: "thisServer"` and `includeContext: "allServers"`, deprecated in the 2025-11-25 revision; the migration is to omit the field or use `"none"`, and earliest removal follows Sampling's own timeline rather than an independent date. The second is the HTTP+SSE transport from 2024-11-05, deprecated in 2025-03-26, whose migration path is Streamable HTTP and whose earliest removal is three months after SEP-2596 reaches Final — the nearest-term removal in the registry. Code that opens a long-lived SSE connection plus a separate POST endpoint is implementing the deprecated HTTP+SSE binding, not the Streamable HTTP binding this revision defines.


## The io.modelcontextprotocol/tasks extension identifier and where the Tasks spec lives

MCP Tasks is no longer part of the core protocol — as of revision 2026-07-28 it is an optional extension identified by `io.modelcontextprotocol/tasks`, and the normative text lives in the `modelcontextprotocol/experimental-ext-tasks` repository rather than in the specification tree. Code written against earlier drafts that assumed every server could hand back a task, or that treated task support as implied by the protocol version, is wrong. Support is negotiated with the standard extension mechanism — the client puts `io.modelcontextprotocol/tasks` in the `extensions` object of the `io.modelcontextprotocol/clientCapabilities` it sends in each request's `_meta`, and the server advertises the same identifier in the `capabilities.extensions` it returns from `server/discover`. Task support requires explicit opt-in from both sides. The extension's settings object is empty — it carries no configuration today.

## Declaring the tasks capability, and what a server must check before returning a CreateTaskResult

**What a server checks before answering with a task**: before a server may respond with a `CreateTaskResult` it must confirm the client opted in, and it finds that in the request's own `_meta` — specifically an `extensions` object naming `io.modelcontextprotocol/tasks` inside `_meta["io.modelcontextprotocol/clientCapabilities"]`. No opt-in there means the server must answer inline; it may not hand back a task the client never agreed to poll for.

Stated from the other side:

A client willing to receive tasks advertises it per request, not once at connection setup. It puts an `extensions` object containing `io.modelcontextprotocol/tasks` inside `_meta["io.modelcontextprotocol/clientCapabilities"]` on the params of every request that could become long-running, such as `tools/call`. The settings value is an empty object. Because this is per-request rather than a handshake flag, there is no per-tool warmup and no per-request "make this async" boolean — the client opts in once in its capability block and the server decides, per request, whether to answer synchronously or with a task.

## CreateTaskResult with resultType task: the handle a server returns instead of blocking

When a server decides a request will take too long to answer inline, it responds with a `CreateTaskResult`, identified by `resultType` set to the string `"task"`. That result carries a `Task` object with a unique `taskId`, the initial status, a `ttlMs` giving how long the task record is retained, and a `pollIntervalMs` giving the suggested polling cadence. The task must be durably created before the response is sent, so the handle is valid the moment the client sees it. The `taskId` is a durable handle rather than an in-memory registration: if the client disconnects, crashes, or restarts, it resumes with the same `taskId`, which is the whole reason Tasks exists rather than holding a connection open past client and proxy timeouts.

## tasks/get polling loop and the pollIntervalMs cadence

Polling is the default way to follow a task. The client calls `tasks/get` with the `taskId` from the `CreateTaskResult`, and the response is the current `Task` object carrying its status. Clients should respect `pollIntervalMs` rather than polling as fast as they like, and continue until the task reaches a terminal status. On `completed`, the `result` field holds exactly what the original request — for example the `tools/call` — would have returned synchronously; on `failed`, the `error` field holds the JSON-RPC error. There is no separate result-fetch call: the final payload is delivered inside the `tasks/get` response itself.

## Task status values: working, input_required, completed, failed, cancelled

A task's status is one of exactly five values. `working` means the operation is in progress. `input_required` means the server needs client input before continuing and the client should read `inputRequests`. `completed` means it finished and the `result` field contains the final output. `failed` means a JSON-RPC error occurred and the `error` field has details. `cancelled` means the operation was cancelled, which is not always honored. `completed`, `failed`, and `cancelled` are terminal — once reached the state does not change, so a client can stop polling and cache the outcome.

## Handling input_required on a task: the inputRequests map and tasks/update

Mid-flight interaction is what makes Tasks usable for approval gates and confirmations. When a running task needs something from the user, it moves to status `input_required`, and the `tasks/get` response then includes an `inputRequests` map holding the outstanding elicitations or other server-initiated requests. The client presents these and submits answers by calling `tasks/update` with `inputResponses` keyed to the entries in `inputRequests`. The server acknowledges `tasks/update` with an empty result, and should ignore responses whose keys are unknown or already satisfied. The design point is that no second connection and no unsolicited server-to-client message is needed — the server's request rides back on the client's own poll.

## tasks/cancel is cooperative, not a guaranteed stop

A client may send `tasks/cancel` with a `taskId` at any point. The server acknowledges with an empty result, but acknowledgement is not a promise: cancellation is explicitly cooperative and the server is not obligated to stop the work. A cancelled-requested task may still finish and land in a non-`cancelled` terminal status such as `completed` or `failed`, so a client must not assume a successful acknowledgement means the side effects were prevented. Keep polling after cancelling if the terminal status matters.

## notifications/tasks push updates and opting in via subscriptions/listen

Servers may push task status changes instead of forcing clients to poll, using the notification method `notifications/tasks`. Clients opt into receiving them through the `subscriptions/listen` mechanism rather than any tasks-specific subscribe call. Each `notifications/tasks` message carries the full task state, so the client does not need a follow-up `tasks/get` round-trip to learn what changed. Polling remains the default and the baseline every implementation must handle; notifications are an optimization a client can rely on only when the server supports them.

## Never return a task to a client that did not declare tasks support

Before responding with a `CreateTaskResult`, a server must check that the incoming request actually declared `io.modelcontextprotocol/tasks` in its per-request `io.modelcontextprotocol/clientCapabilities` extensions object. The instruction is unambiguous: never return a task to a client that did not declare support. A client without the extension has no `tasks/get` implementation and will fail to interpret a `resultType: "task"` payload, so the server must either do the work synchronously or return an error. This is the tasks-specific instance of the general rule that extensions are disabled by default and require explicit opt-in.

## Methods defined by the Tasks extension: only tasks/get, tasks/update, tasks/cancel

The task surface is small and worth memorizing, because models trained on earlier drafts confidently emit method names that no longer exist. The client-callable methods are `tasks/get` (poll current state and retrieve the final `result` or `error`), `tasks/update` (submit `inputResponses` for an `input_required` task), and `tasks/cancel` (request cooperative cancellation). The one notification is `notifications/tasks`. Task creation has no method of its own — a task is created as the response to an ordinary request such as `tools/call`. Nothing defines a separate call to fetch a completed task's result, and nothing defines a call to enumerate outstanding tasks; the client is told to persist `taskId` values itself.

## Client obligations for tasks: polymorphic results and durably stored task IDs

Two client-side requirements are easy to miss. First, once a client declares the tasks extension, every supported request becomes polymorphic: a `tools/call` may return the ordinary result or a `CreateTaskResult` with `resultType: "task"`, and the client must branch on `resultType` rather than assuming the synchronous shape. Second, the client must persist task IDs durably, to disk or equivalent, so polling can resume after a crash or restart — there is no server-side listing endpoint to recover a forgotten `taskId` from, so losing the ID loses the task.

## MCP extension identifier format and reverse-DNS vendor prefixes

Every MCP extension is named by a unique identifier of the form vendor-prefix followed by a slash and the extension name, for example `io.modelcontextprotocol/oauth-client-credentials`. Identifiers follow the same rules as `_meta` keys, except the prefix is mandatory rather than optional. Official extensions use the `io.modelcontextprotocol` vendor prefix. Third-party authors should use a reversed domain name they own, in the manner of Java package naming — a company owning `example.com` publishes `com.example/my-extension` — so independently developed extensions cannot collide in the shared `extensions` namespace.

## Official MCP extension identifiers: ui, oauth-client-credentials, enterprise-managed-authorization

The published identifier table names three official extensions. MCP Apps is `io.modelcontextprotocol/ui` — note the identifier says `ui`, not `apps`, even though the docs and repository are called MCP Apps, and this mismatch is a common source of wrong capability keys. OAuth Client Credentials is `io.modelcontextprotocol/oauth-client-credentials`, for machine-to-machine auth without an interactive login. Enterprise-Managed Authorization is `io.modelcontextprotocol/enterprise-managed-authorization`, for centralized access control through an enterprise IdP. MCP Tasks uses `io.modelcontextprotocol/tasks` and is documented alongside these but still lives in an experimental repository.

## Advertising extensions from a server in the server/discover response

A server declares which extensions it supports in the `capabilities.extensions` object of its `server/discover` result, alongside core capability entries such as `tools`. The surrounding result also carries `resultType: "complete"`, a `supportedVersions` array, a `_meta` block holding `io.modelcontextprotocol/serverInfo`, plus `ttlMs` and `cacheScope`. Each extension entry's value is its settings object, whose schema the extension itself defines; an empty object indicates the extension takes no settings.

```json
"capabilities": {"tools": {},
  "extensions": {"io.modelcontextprotocol/ui": {}}}
```

## Extensions are off by default and each SDK chooses which to implement

Extensions are always disabled by default and require explicit opt-in from the developer — a library that ships extension code does not thereby turn it on. SDKs may choose to implement extensions, but implementing any of them is not required for protocol conformance, and SDK maintainers have full autonomy over which they support. The practical consequence for a server author is that you cannot infer extension support from an SDK version or a protocol version — the only reliable signal is the `extensions` object the peer actually sent.

## Versioning an MCP extension: settings flags versus a new -v2 identifier

Extensions evolve independently of the core protocol; updates are managed by the extension repository maintainers and do not require core maintainer review. When changing an extension, prefer capability flags or versioning inside the settings object over minting a new identifier. Only if a breaking change is unavoidable should you publish a new identifier, spelled like `io.modelcontextprotocol/my-extension-v2`. A breaking change is any modification causing existing implementations to fail or behave incorrectly, and the list is explicit: removing or renaming fields, changing field types, altering the semantics of existing behavior, and adding new required fields.

## Official ext- repositories versus incubating experimental-ext- repositories

Official extensions live in the modelcontextprotocol GitHub organization in repositories prefixed `ext-`, such as `ext-auth` and `ext-apps`. Experimental extensions use the prefix `experimental-ext-`, for example `experimental-ext-interceptors` and `experimental-ext-tasks`. The experimental track is an incubation pathway for Working Groups and Interest Groups to prototype before formal SEP submission: every experimental extension must be tied to a working or interest group, its repositories and packages must clearly mark their experimental status, and Core Maintainers retain the right to archive or remove them. Promotion to official status goes through the SEP process on the Extensions Track, which also requires at least one reference implementation in an official SDK.

## Declaring MCP Apps support with io.modelcontextprotocol/ui and the mcp-app MIME type

MCP Apps lets a server return an interactive HTML interface — a chart, form, dashboard, or media viewer — that renders inline in the conversation instead of a wall of text. Its extension identifier is `io.modelcontextprotocol/ui`. Unlike the tasks extension its client-side settings object is not empty: the client declares a `mimeTypes` array, and the value used for MCP App payloads is `text/html;profile=mcp-app`. A server advertises the same key in its `server/discover` capabilities. Because support varies widely by host, a server offering UI-enhanced tools should still return meaningful text content for clients that do not declare the extension.

## Pointing a tool at its app with _meta.ui.resourceUri and a ui:// resource

An MCP App is wired up by combining two existing primitives rather than inventing a new one: a tool declares a reference to a UI resource in its description via `_meta.ui.resourceUri`, whose value is a `ui://` resource URI, and the server exposes that URI as a resource whose body is an HTML page, usually bundled with its JavaScript and CSS. Because the reference lives in the tool description, the host can fetch and preload the app before the tool is ever called, which is what enables features such as streaming tool inputs into a live app. At call time the host renders the fetched HTML in place and pushes the tool result into it.

## Controlling an app's origins and device access with _meta.ui.csp and _meta.ui.permissions

The UI resource's `_meta.ui` object carries two policy fields beyond the URI. `csp` controls which external origins the app may load scripts and other resources from — an app is not restricted to fully self-contained bundles, but any external origin it uses must be listed there. `permissions` requests additional host capabilities the app needs, such as microphone or camera access. Both are requests to the host, which remains the enforcement point; a host may restrict which tools an app is permitted to call and may disable individual capabilities regardless of what the app asks for.

## The app-to-host ui/ JSON-RPC dialect over postMessage

An MCP App does not speak the ordinary MCP transport. It communicates with the host over a JSON-RPC protocol that is its own dialect of MCP, carried by the browser `postMessage` API rather than stdio or Streamable HTTP. Some messages are shared verbatim with core MCP — `tools/call` is the same method — some are analogous but renamed, such as `ui/initialize`, and most app-specific messages are new and carry the `ui/` method prefix. Through this channel the app can request tool calls, send messages, update the model's context, and receive pushed data from the host. Because it is plain web primitives, any framework or none will do.

## MCP Apps sandbox: what the iframe blocks and why hosts can render untrusted servers

MCP Apps run inside a sandboxed iframe controlled by the host, and the isolation is what lets a host render an app from a server whose author it does not fully trust. The sandbox prevents the app from accessing the parent window's DOM, reading the host's cookies or local storage, navigating the parent page, or executing scripts in the parent context. All communication goes through `postMessage`, so the host mediates every privileged action — including which tools the app may call. A useful consequence for design: an app should ask the host for an outcome, such as "schedule this meeting," and let the host route it through capabilities the user has already connected, instead of building its own direct integrations and credentials.

## Choosing between the oauth-client-credentials and enterprise-managed-authorization extensions

The core specification's OAuth 2.0 authorization-code framework already covers the common case of a user interactively granting a client access, and needs no extension. Two official extensions in the `ext-auth` repository cover what it does not. Use `io.modelcontextprotocol/oauth-client-credentials`, the OAuth 2.0 client credentials flow, when there is no human in the loop: background services and daemons, CI/CD pipelines calling MCP tools, and server-to-server integrations. Use `io.modelcontextprotocol/enterprise-managed-authorization` when a centralized identity provider must enforce policy — employees reaching MCP servers through their organization's IdP, or organization-wide access policy enforcement. Both require explicit client support and are never active by default.

## Which clients ship MCP Apps support today, and why the auth columns are blank

The community-maintained extension support matrix lists MCP Apps as supported by Claude on the web, Claude Desktop, VS Code GitHub Copilot, Microsoft 365 Copilot, Goose, Postman, MCPJam, ChatGPT, Cursor, Archestra.AI, and PostHog Code. For the two auth extensions the matrix is nearly empty: Archestra.AI is the only client marked as supporting Enterprise-Managed Authorization, and no listed client is marked for OAuth Client Credentials. The matrix warns that auth extension support is tracked separately from the core MCP authorization features such as DCR and CIMD, so a client implementing those still shows blank. MCP Tasks has no column at all — consistent with it still living in an experimental repository.

