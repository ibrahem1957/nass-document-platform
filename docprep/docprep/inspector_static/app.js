"use strict";

const state = {
  documents: [], currentId: null, payload: null, tab: "elements", page: 1,
  selectedSection: null, selectedElement: null, maps: {},
};

const $ = (id) => document.getElementById(id);
const ui = {
  documentSelect: $("documentSelect"), status: $("status"), overview: $("overview"),
  sectionTree: $("sectionTree"), showAllElements: $("showAllElements"),
  previousPage: $("previousPage"), nextPage: $("nextPage"), pageInput: $("pageInput"),
  pageTotal: $("pageTotal"), zoomSelect: $("zoomSelect"), pdfFrame: $("pdfFrame"),
  pdfFallback: $("pdfFallback"), geometryNotice: $("geometryNotice"), tabs: $("tabs"),
  searchPanel: $("searchPanel"), searchInput: $("searchInput"), searchMode: $("searchMode"),
  clearSearch: $("clearSearch"), qualityFilters: $("qualityFilters"),
  severityFilter: $("severityFilter"), reviewFilter: $("reviewFilter"), view: $("view"),
  detailDrawer: $("detailDrawer"), detailTitle: $("detailTitle"), detailBody: $("detailBody"),
  closeDetails: $("closeDetails"),
};

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined && text !== null) value.textContent = String(text);
  return value;
}

function badge(text, kind = "") {
  return node("span", `badge ${kind}`.trim(), text);
}

function setStatus(message, error = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", error);
}

async function fetchJson(url) {
  const response = await fetch(url, {cache: "no-store", credentials: "same-origin"});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function appendDefinitionList(parent, rows) {
  const list = node("dl", "metadata-grid");
  rows.forEach(([term, value]) => {
    list.append(node("dt", "", term), node("dd", "", value === null || value === undefined ? "Unavailable" : value));
  });
  parent.append(list);
}

function appendJson(parent, value, className = "evidence") {
  const block = node("pre", className);
  block.textContent = JSON.stringify(value, null, 2);
  parent.append(block);
}

function highlightedText(parent, text, query) {
  const original = text === null || text === undefined ? "" : String(text);
  if (!query) {
    parent.textContent = original;
    return;
  }
  const lower = original.toLocaleLowerCase();
  const needle = query.toLocaleLowerCase();
  let cursor = 0;
  let found = lower.indexOf(needle);
  while (found >= 0) {
    parent.append(document.createTextNode(original.slice(cursor, found)));
    parent.append(node("mark", "", original.slice(found, found + needle.length)));
    cursor = found + needle.length;
    found = lower.indexOf(needle, cursor);
  }
  parent.append(document.createTextNode(original.slice(cursor)));
}

function buildMaps(documentData) {
  const by = (items, key) => new Map(items.map((item) => [item[key], item]));
  return {
    elements: by(documentData.elements, "element_id"), sections: by(documentData.sections, "section_id"),
    relationships: by(documentData.relationships, "rel_id"),
    findings: by(documentData.quality.findings, "finding_id"), visuals: by(documentData.visuals, "visual_id"),
    logicalTables: by(documentData.logical_tables, "logical_table_id"), nodes: by(documentData.nodes, "node_id"),
    lists: by(documentData.lists, "list_id"),
  };
}

async function initialize() {
  try {
    const response = await fetchJson("/api/documents");
    state.documents = response.documents;
    if (!state.documents.length) throw new Error("No validated documents are configured");
    state.documents.forEach((documentSummary) => {
      const option = node("option", "", documentSummary.label);
      option.value = documentSummary.id;
      ui.documentSelect.append(option);
    });
    await loadDocument(state.documents[0].id);
  } catch (error) {
    setStatus(`Inspector unavailable: ${error.message}`, true);
  }
}

async function loadDocument(documentId) {
  setStatus("Loading validated prepared document…");
  try {
    const payload = await fetchJson(`/api/documents/${encodeURIComponent(documentId)}`);
    state.currentId = documentId;
    state.payload = payload;
    state.page = 1;
    state.selectedSection = null;
    state.selectedElement = null;
    state.maps = buildMaps(payload.document);
    ui.documentSelect.value = documentId;
    ui.searchInput.value = "";
    ui.reviewFilter.checked = false;
    renderOverview();
    renderSectionTree();
    configureSeverityFilter();
    setPage(1);
    renderCurrentView();
    const pdfMessage = payload.summary.pdf_available ? "PDF verified and available locally" : payload.summary.pdf_status;
    setStatus(`${payload.summary.label} loaded · ${pdfMessage}`);
  } catch (error) {
    setStatus(`Unable to load document: ${error.message}`, true);
  }
}

function renderOverview() {
  const summary = state.payload.summary;
  ui.overview.replaceChildren();
  const metrics = [
    [summary.title || summary.label, "Document"], [summary.page_count, "Pages"],
    [summary.element_count, "Elements"], [summary.section_count, "Sections"],
    [summary.relationship_count, "Relationships"], [summary.quality_finding_count, "Quality findings"],
  ];
  metrics.forEach(([value, label]) => {
    const item = node("div", "metric");
    item.append(node("strong", "", value), node("span", "", label));
    item.title = label === "Document" ? (summary.source_filename || "") : "";
    ui.overview.append(item);
  });
}

function renderSectionTree() {
  ui.sectionTree.replaceChildren();
  const sections = state.payload.document.sections;
  if (!sections.length) {
    ui.sectionTree.append(node("p", "empty-state", "No sections were detected. Elements remain available in reading order."));
    return;
  }
  const roots = sections.filter((section) => section.parent === null || !state.maps.sections.has(section.parent));
  const visited = new Set();
  roots.forEach((section) => ui.sectionTree.append(sectionBranch(section, visited)));
  sections.filter((section) => !visited.has(section.section_id)).forEach((section) => ui.sectionTree.append(sectionBranch(section, visited)));
}

function sectionBranch(section, visited) {
  visited.add(section.section_id);
  const branch = node("div", "section-branch");
  const button = node("button", `section-button${state.selectedSection === section.section_id ? " selected" : ""}`);
  button.type = "button";
  button.append(document.createTextNode(section.title || "Untitled section"), node("small", "", `${section.section_id} · level ${section.level} · pages ${section.page_start}–${section.page_end}`));
  button.addEventListener("click", () => selectSection(section.section_id));
  branch.append(button);
  (section.children || []).forEach((childId) => {
    const child = state.maps.sections.get(childId);
    if (child && !visited.has(childId)) branch.append(sectionBranch(child, visited));
  });
  return branch;
}

function selectSection(sectionId) {
  const section = state.maps.sections.get(sectionId);
  if (!section) return;
  state.selectedSection = sectionId;
  state.tab = "elements";
  setActiveTab();
  renderSectionTree();
  setPage(section.page_start);
  renderCurrentView();
}

function setPage(requested) {
  const maximum = state.payload ? state.payload.summary.page_count : 1;
  const parsed = Number.parseInt(requested, 10);
  state.page = Math.max(1, Math.min(maximum, Number.isFinite(parsed) ? parsed : 1));
  ui.pageInput.max = String(maximum);
  ui.pageInput.value = String(state.page);
  ui.pageTotal.textContent = `of ${maximum}`;
  updatePdf();
}

function updatePdf() {
  if (!state.payload) return;
  const summary = state.payload.summary;
  if (!summary.pdf_available) {
    ui.pdfFrame.hidden = true;
    ui.pdfFrame.removeAttribute("src");
    ui.pdfFallback.hidden = false;
    ui.pdfFallback.textContent = `PDF preview unavailable: ${summary.pdf_status}. Structured inspection remains available.`;
  } else {
    ui.pdfFallback.hidden = true;
    ui.pdfFrame.hidden = false;
    ui.pdfFrame.src = `/api/documents/${encodeURIComponent(state.currentId)}/pdf#page=${state.page}&zoom=${encodeURIComponent(ui.zoomSelect.value)}`;
  }
  updateGeometryNotice();
}

function updateGeometryNotice() {
  const selected = state.selectedElement ? state.maps.elements.get(state.selectedElement) : null;
  if (!selected) {
    ui.geometryNotice.textContent = "Select an element to synchronize its source page. No overlay is inferred.";
    return;
  }
  const reliable = selected.geometry_reliability === "precise" || selected.geometry_reliability === "approximate";
  ui.geometryNotice.textContent = reliable && selected.bbox_pt
    ? `${selected.element_id}: ${selected.geometry_reliability} saved geometry ${JSON.stringify(selected.bbox_pt)}. Page synchronized; native viewer does not draw overlays.`
    : `${selected.element_id}: ${selected.geometry_reliability} geometry. Page synchronized without a bounding box.`;
}

function setActiveTab() {
  ui.tabs.querySelectorAll("button[data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.tab));
  ui.searchPanel.hidden = state.tab !== "elements";
  ui.qualityFilters.hidden = state.tab !== "quality";
}

function renderCurrentView() {
  if (!state.payload) return;
  setActiveTab();
  ui.view.replaceChildren();
  if (state.tab === "elements") renderElements();
  else if (state.tab === "relationships") renderRelationships();
  else if (state.tab === "quality") renderQuality();
  else if (state.tab === "visuals") renderVisuals();
  else renderJson();
}

function matchesElement(element) {
  if (state.selectedSection && !(element.section_ids || []).includes(state.selectedSection)) return false;
  const query = ui.searchInput.value.trim().toLocaleLowerCase();
  if (!query) return true;
  const mode = ui.searchMode.value;
  const sectionText = (element.section_ids || []).map((sectionId) => {
    const section = state.maps.sections.get(sectionId);
    return `${sectionId} ${section?.title || ""}`;
  }).join(" ");
  const values = {
    id: element.element_id, text: element.text || "", page: String(element.page),
    section: sectionText,
  };
  if (mode === "page") return values.page === query;
  if (mode === "all") return Object.values(values).some((value) => value.toLocaleLowerCase().includes(query));
  return values[mode].toLocaleLowerCase().includes(query);
}

function renderElements() {
  const query = ui.searchInput.value.trim();
  const elements = state.payload.document.elements.filter(matchesElement);
  ui.view.append(node("p", "result-count", `${elements.length} element(s) in original reading order${elements.length > 300 ? " · first 300 shown" : ""}`));
  elements.slice(0, 300).forEach((element) => {
    const card = node("article", "card clickable");
    const header = node("div", "card-header");
    const heading = node("h3", "");
    highlightedText(heading, element.element_id, ui.searchMode.value === "id" || ui.searchMode.value === "all" ? query : "");
    const badges = node("div", "badges");
    badges.append(badge("Original evidence", "original"), badge(element.original_type), badge(`Page ${element.page}`), badge(element.validation_state || "unknown", element.validation_state || ""));
    header.append(heading, badges);
    const text = node("pre", "source-text");
    text.dir = "auto";
    highlightedText(text, element.text, ui.searchMode.value === "text" || ui.searchMode.value === "all" ? query : "");
    card.append(header, text);
    appendDefinitionList(card, [["Sections", (element.section_ids || []).join(", ") || "None"], ["Roles", (element.roles || []).join(", ") || "None"], ["Source provenance", element.source_extraction?.source || "Unavailable"], ["Geometry", element.geometry_reliability], ["Relationships", (element.relationship_ids || []).join(", ") || "None"], ["Quality findings", (element.quality_finding_ids || []).join(", ") || "None"]]);
    card.addEventListener("click", () => selectElement(element.element_id));
    ui.view.append(card);
  });
}

function selectElement(elementId) {
  const element = state.maps.elements.get(elementId);
  if (!element) return;
  state.selectedElement = elementId;
  setPage(element.page);
  showDetails(`Element ${elementId}`, element, "Original source element");
}

function endpointButton(nodeId) {
  const button = node("button", "endpoint", nodeId);
  button.type = "button";
  button.addEventListener("click", () => selectNode(nodeId));
  return button;
}

function renderRelationships() {
  const relationships = state.payload.document.relationships;
  ui.view.append(node("p", "result-count", `${relationships.length} relationship(s); confidence and review status are preserved.`));
  relationships.forEach((relationship) => {
    const card = node("article", "card");
    const header = node("div", "card-header");
    header.append(node("h3", "", `${relationship.type} · ${relationship.rel_id}`));
    const badges = node("div", "badges");
    badges.append(badge("Inferred relationship", "derived"), badge(relationship.confidence, relationship.confidence));
    if (relationship.review_recommended) badges.append(badge("Needs review", "needs_review"));
    header.append(badges);
    const endpoints = node("div", "relationship-endpoints");
    endpoints.append(endpointButton(relationship.source_id), node("span", "connector", "→"), endpointButton(relationship.target_id));
    card.append(header, endpoints);
    appendDefinitionList(card, [["Method", relationship.method], ["Limitations", (relationship.limitations || []).join("; ") || "None"], ["Conflicts", (relationship.conflicts || []).length]]);
    const evidenceTitle = node("strong", "", "Evidence");
    card.append(evidenceTitle);
    appendJson(card, relationship.evidence);
    ui.view.append(card);
  });
}

function selectNode(nodeId) {
  const element = state.maps.elements.get(nodeId);
  if (element) {
    selectElement(nodeId);
    return;
  }
  const section = state.maps.sections.get(nodeId);
  if (section) {
    selectSection(nodeId);
    showDetails(`Section ${nodeId}`, section, "Derived structure");
    return;
  }
  const logicalTable = state.maps.logicalTables.get(nodeId);
  if (logicalTable) {
    setPage(logicalTable.pages[0]);
    showDetails(`Logical table ${nodeId}`, logicalTable, "Derived table grouping");
    return;
  }
  const virtualNode = state.maps.nodes.get(nodeId) || state.maps.lists.get(nodeId);
  showDetails(`Node ${nodeId}`, virtualNode || {node_id: nodeId, status: "Reference resolves upstream but has no display record"}, "Virtual or derived node");
}

function configureSeverityFilter() {
  const severities = Array.from(new Set(state.payload.document.quality.findings.map((finding) => finding.severity)));
  ui.severityFilter.replaceChildren();
  const all = node("option", "", "All"); all.value = "all"; ui.severityFilter.append(all);
  severities.forEach((severity) => { const option = node("option", "", severity); option.value = severity; ui.severityFilter.append(option); });
}

function renderQuality() {
  const severity = ui.severityFilter.value;
  const needsReview = ui.reviewFilter.checked;
  const findings = state.payload.document.quality.findings.filter((finding) => {
    const severityMatch = severity === "all" || finding.severity === severity;
    const reviewMatch = !needsReview || finding.status === "needs_review" || finding.source_validation_states?.includes("needs_review");
    return severityMatch && reviewMatch;
  });
  ui.view.append(node("p", "result-count", `${findings.length} finding(s). Low-confidence and unverified findings are not presented as confirmed errors.`));
  findings.forEach((finding) => {
    const card = node("article", "card");
    const header = node("div", "card-header");
    header.append(node("h3", "", `${finding.category} / ${finding.subtype}`));
    const badges = node("div", "badges");
    badges.append(badge(finding.severity, finding.severity), badge(`${finding.confidence} confidence`, finding.confidence), badge(finding.status, finding.status));
    header.append(badges);
    card.append(header);
    appendDefinitionList(card, [["Finding ID", finding.finding_id], ["Pages", (finding.page_numbers || []).join(", ") || "Unknown"], ["Affected nodes", (finding.source_ids || []).join(", ") || "None"], ["Explanation", finding.explanation], ["Suggested action", finding.suggested_review_action]]);
    (finding.source_ids || []).forEach((sourceId) => card.append(endpointButton(sourceId), document.createTextNode(" ")));
    card.append(node("strong", "", "Evidence"));
    appendJson(card, finding.evidence);
    ui.view.append(card);
  });
}

function renderVisuals() {
  const documentData = state.payload.document;
  ui.view.append(node("p", "result-count", `${documentData.visuals.length} figure/table element(s) and ${documentData.logical_tables.length} logical table grouping(s).`));
  documentData.logical_tables.forEach((table) => {
    const card = node("article", "card");
    const header = node("div", "card-header");
    header.append(node("h3", "", table.logical_table_id));
    const badges = node("div", "badges"); badges.append(badge("Logical multi-page table", "derived")); header.append(badges);
    card.append(header);
    appendDefinitionList(card, [["Pages", table.pages.join(" → ")], ["Ordered source parts", table.source_part_ids.join(" → ")], ["Repeated header evidence", table.repeated_header_between_parts.join(", ")], ["Quality findings", table.quality_finding_ids.length]]);
    table.source_part_ids.forEach((id) => card.append(endpointButton(id), document.createTextNode(" ")));
    ui.view.append(card);
  });
  documentData.visuals.forEach((visual) => {
    const card = node("article", "card clickable");
    const header = node("div", "card-header");
    header.append(node("h3", "", `${visual.visual_type} · ${visual.visual_id}`));
    const badges = node("div", "badges");
    badges.append(badge("Original evidence", "original"), badge(`Page ${visual.page}`), badge(visual.validation_state || "unknown", visual.validation_state || ""));
    header.append(badges);
    const content = node("pre", "source-text"); content.dir = "auto"; content.textContent = visual.original_content || "No extracted source text";
    card.append(header, content);
    appendDefinitionList(card, [["Section", visual.section_id || "Unknown"], ["Caption", visual.caption ? JSON.stringify(visual.caption) : "Unknown"], ["Description paragraphs", (visual.description_paragraphs || []).join(", ") || "None"], ["Relationships", (visual.relationship_ids || []).join(", ") || "None"], ["Quality findings", (visual.quality_finding_ids || []).join(", ") || "None"], ["Logical table", visual.logical_table_id || "No"]]);
    card.addEventListener("click", () => { setPage(visual.page); showDetails(`Visual ${visual.visual_id}`, visual, "Original visual/table inventory"); });
    ui.view.append(card);
  });
}

function renderJson() {
  const note = node("p", "result-count", "Exact read-only prepared_document.json values. Source strings are not normalized.");
  const pre = node("pre", "json-view");
  pre.textContent = JSON.stringify(state.payload.document, null, 2);
  ui.view.append(note, pre);
}

function showDetails(title, value, classification) {
  ui.detailTitle.textContent = title;
  ui.detailBody.replaceChildren();
  ui.detailBody.append(badge(classification, classification.startsWith("Original") ? "original" : "derived"));
  appendJson(ui.detailBody, value, "json-view");
  ui.detailDrawer.hidden = false;
}

ui.documentSelect.addEventListener("change", () => loadDocument(ui.documentSelect.value));
ui.previousPage.addEventListener("click", () => setPage(state.page - 1));
ui.nextPage.addEventListener("click", () => setPage(state.page + 1));
ui.pageInput.addEventListener("change", () => setPage(ui.pageInput.value));
ui.zoomSelect.addEventListener("change", updatePdf);
ui.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tab]");
  if (!button) return;
  state.tab = button.dataset.tab;
  renderCurrentView();
});
ui.searchInput.addEventListener("input", renderCurrentView);
ui.searchMode.addEventListener("change", renderCurrentView);
ui.clearSearch.addEventListener("click", () => { ui.searchInput.value = ""; renderCurrentView(); });
ui.severityFilter.addEventListener("change", renderCurrentView);
ui.reviewFilter.addEventListener("change", renderCurrentView);
ui.showAllElements.addEventListener("click", () => { state.selectedSection = null; state.tab = "elements"; renderSectionTree(); renderCurrentView(); });
ui.closeDetails.addEventListener("click", () => { ui.detailDrawer.hidden = true; });

initialize();
