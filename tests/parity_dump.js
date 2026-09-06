#!/usr/bin/env node
// Dumps the output of the original Node backend's pure helper functions for a fixed set of
// inputs, so test_parity.py can assert the Python port produces byte-identical results.
// Points at the DevAgents client this service was ported from; override with
// SUBJECTIVE_DEVAGENTS_CLIENT_ROOT.
'use strict';

const path = require('path');

const CLIENT_ROOT = process.env.SUBJECTIVE_DEVAGENTS_CLIENT_ROOT
  || path.resolve(__dirname, '..', '..', '..', 'clients', 'desktop', 'subjective_devagents');
const llm = require(path.join(CLIENT_ROOT, 'src', 'llmService.js'));
const users = require(path.join(CLIENT_ROOT, 'src', 'systemUsers.js'));

const cases = JSON.parse(require('fs').readFileSync(path.join(__dirname, 'parity_cases.json'), 'utf8'));

const out = {
  providerLabel: cases.providers.map((value) => llm.providerLabel(value)),
  providerIcon: cases.providers.map((value) => llm.providerIcon(value)),
  normalizeRole: cases.roles.map((value) => llm.normalizeRole(value)),
  epochMs: cases.times.map((value) => llm.epochMs(value)),
  exportFilenameTime: cases.exportPaths.map((value) => llm.exportFilenameTime(value)),
  cleanVisibleMessage: cases.messages.map((value) => llm.cleanVisibleMessage(value, 'user')),
  isGenericProviderLabel: cases.labelPairs.map(([label, provider]) => llm.isGenericProviderLabel(label, provider)),
  preferConversationLabel: cases.labelTriples.map(([existing, incoming, provider]) => (
    llm.preferConversationLabel(existing, incoming, provider)
  )),
  ircChannelName: cases.channelInputs.map(([label, identity, provider]) => llm.ircChannelName(label, identity, provider)),
  composerIdFromSource: cases.sourceFiles.map((value) => llm.composerIdFromSource(value)),
  sourceUserFromPath: cases.sourceFiles.map((value) => llm.sourceUserFromPath(value)),
  sessionIdentity: cases.identityItems.map((value) => llm.sessionIdentity(value)),
  formatTranscriptForHandoff: cases.transcripts.map((value) => llm.formatTranscriptForHandoff(value)),
  tailConversationForClient: cases.tailConversations.map((value) => llm.tailConversationForClient(value)),
  isValidUsername: cases.usernames.map((value) => users.isValidUsername(value)),
  parsePasswdText: users.parsePasswdText(cases.passwdText),
  filterHumanUsers: users.filterHumanUsers(users.parsePasswdText(cases.passwdText), { excludeUsername: 'bob' })
};

process.stdout.write(JSON.stringify(out, null, 2));
