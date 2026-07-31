# mcp-spec constitution

Applied to every MCP client or server written while this pack is mounted.
The target is specification revision 2026-07-28. Every rule exists
because model training data teaches the OLD protocol — the handshake,
sessions, and server-initiated requests this revision removed. Terse on
purpose; message shapes and reasoning live in the corpus.

## No initialize handshake

There is no `initialize` / `notifications/initialized` exchange. Every
request is self-contained: `_meta` carries
`io.modelcontextprotocol/protocolVersion` and
`io.modelcontextprotocol/clientCapabilities` (both required). Writing a
handshake means writing the 2025 protocol.

## No sessions

Never mint, expect, or echo `Mcp-Session-Id`. State that spans requests
travels as explicit server-minted handles passed as ordinary tool
arguments, integrity-protected if they influence authorization.

## Servers never initiate requests

`sampling/createMessage`, `elicitation/create` and `roots/list` are sent
only inside an `InputRequiredResult` (`resultType: "input_required"`)
with `inputRequests`; the client retries the original request with
`inputResponses` and a NEW JSON-RPC id, echoing `requestState`
untouched. Only `tools/call`, `resources/read` and `prompts/get` may
return it.

## Every result carries resultType

`"complete"` for ordinary results. Treat an absent `resultType` from an
older server as `"complete"`; treat an unrecognized value as invalid.

## HTTP requests carry the mirrored headers

Single POST endpoint. Every POST includes `MCP-Protocol-Version`,
`Mcp-Method`, and `Mcp-Name` (for tools/call, resources/read,
prompts/get), matching the body exactly — mismatch is `400` +
`HeaderMismatch` (-32020). No GET stream, no `Last-Event-ID` resume: a
broken stream means re-issue the request with a new id.

## Do not implement Roots, Sampling, or Logging in new code

All three are deprecated. Pass directories as tool parameters or
configuration; call LLM provider APIs directly instead of sampling; log
to stderr (stdio) or OpenTelemetry. Per-request log level comes from
`io.modelcontextprotocol/logLevel` in `_meta` — never emit
`notifications/message` for a request that did not include it.

## Change notifications go through subscriptions/listen

One long-lived request whose response stream carries only opted-in types
(`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`,
`resourceSubscriptions`). Acknowledge first; tag everything with
`io.modelcontextprotocol/subscriptionId`. `resources/subscribe`,
`resources/unsubscribe` and `ping` no longer exist.

## Use the error codes as allocated

Resource not found is `-32602`, not `-32002`. `-32020` HeaderMismatch,
`-32021` MissingRequiredClientCapability, `-32022`
UnsupportedProtocolVersion. Never invent codes in `-32020`..`-32099`;
put application errors outside `-32768`..`-32000`.

## requestState is attacker-controlled

If it influences authorization, resource access or business logic:
HMAC or AEAD, verify principal, TTL, and originating-request digest on
receipt, reject on failure. Clients never inspect or modify it.

## Implement server/discover

Mandatory for servers: advertise supported versions, capabilities and
identity. Reject unsupported versions with
`UnsupportedProtocolVersionError` listing `supported`, so clients can
retry with a mutual version.

## Validate Origin, bind localhost

HTTP servers reject invalid `Origin` with 403 (DNS rebinding), bind
127.0.0.1 when local, and authenticate connections.

## State the revision when uncertain

If asked about MCP behaviour this pack does not cover, say the answer is
from model memory and may describe an earlier protocol revision.
