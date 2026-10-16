"""Hierarchical Document Tree Index and Navigation for SEC 10-K Filings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class DocTreeNode:
    """A node in the hierarchical SEC 10-K document tree."""

    node_id: str
    title: str
    level: int  # 0: Doc, 1: Part, 2: Item, 3: Statement/Note, 4: Chunk
    statement_type: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    children: dict[str, "DocTreeNode"] = field(default_factory=dict)

    def add_child(self, child: "DocTreeNode") -> "DocTreeNode":
        self.children[child.node_id] = child
        return child

    def collect_chunk_ids(self) -> list[str]:
        """Collect all chunk_ids under this node recursively."""
        chunks = list(self.chunk_ids)
        for child in self.children.values():
            chunks.extend(child.collect_chunk_ids())
        return chunks


class DocumentHierarchyTree:
    """Builds and queries the hierarchical tree of SEC 10-K document chunks."""

    def __init__(self, doc_id: str = "SEC_10K") -> None:
        self.doc_id = doc_id
        self.root = DocTreeNode(node_id="root", title=f"SEC 10-K Filing: {doc_id}", level=0)
        self._all_chunk_ids: set[str] = set()
        self._init_standard_skeleton()

    def _init_standard_skeleton(self) -> None:
        """Initialize standard SEC 10-K Part and Item structure."""
        part1 = self.root.add_child(DocTreeNode("part_1", "Part I", level=1))
        part1.add_child(DocTreeNode("item_1", "Item 1. Business", level=2))
        part1.add_child(DocTreeNode("item_1a", "Item 1A. Risk Factors", level=2))

        part2 = self.root.add_child(DocTreeNode("part_2", "Part II", level=1))
        part2.add_child(DocTreeNode("item_7", "Item 7. Management's Discussion and Analysis", level=2))
        part2.add_child(DocTreeNode("item_7a", "Item 7A. Quantitative and Qualitative Disclosures About Market Risk", level=2))

        item8 = part2.add_child(DocTreeNode("item_8", "Item 8. Financial Statements and Supplementary Data", level=2))
        item8.add_child(DocTreeNode("item_8_operations", "Consolidated Statements of Operations", level=3, statement_type="income_statement"))
        item8.add_child(DocTreeNode("item_8_balance_sheet", "Consolidated Balance Sheets", level=3, statement_type="balance_sheet"))
        item8.add_child(DocTreeNode("item_8_cash_flows", "Consolidated Statements of Cash Flows", level=3, statement_type="cash_flow"))
        item8.add_child(DocTreeNode("item_8_notes", "Notes to Consolidated Financial Statements", level=3, statement_type="note_disclosures"))

        part3 = self.root.add_child(DocTreeNode("part_3", "Part III", level=1))
        part3.add_child(DocTreeNode("item_10", "Item 10. Directors, Executive Officers and Corporate Governance", level=2))
        part3.add_child(DocTreeNode("item_11", "Item 11. Executive Compensation", level=2))

    @property
    def total_chunks(self) -> int:
        return len(self._all_chunk_ids)

    def index_chunk(self, chunk_id: str, metadata: Mapping[str, Any]) -> None:
        """Map one physical chunk to its location in the hierarchy."""
        self._all_chunk_ids.add(chunk_id)
        section = str(metadata.get("section") or "").lower()
        title = str(metadata.get("title") or metadata.get("table_name") or "").lower()
        text = str(metadata.get("source_text") or metadata.get("text") or "").lower()
        statement_type = str(metadata.get("statement_type") or "").lower()

        target_node = self._locate_node(section, title, text, statement_type)
        target_node.chunk_ids.append(chunk_id)

    def _locate_node(self, section: str, title: str, text: str, statement_type: str) -> DocTreeNode:
        part2 = self.root.children["part_2"]
        item8 = part2.children["item_8"]

        # 1. MD&A in Item 7
        if "item 7" in section or "item 7" in title or "md&a" in title or "discussion and analysis" in text or "discussion and analysis" in title:
            return part2.children["item_7"]

        # 2. Risk Factors in Item 1A
        if "item 1a" in section or "risk factors" in title or "risk factors" in text:
            return self.root.children["part_1"].children["item_1a"]

        # 3. Business in Item 1
        if "item 1" in section or "business" in title:
            return self.root.children["part_1"].children["item_1"]

        # 4. Financial statements and notes in Item 8
        if statement_type == "income_statement" or "statement of operations" in title or "statements of operations" in title or "income statement" in title:
            return item8.children["item_8_operations"]
        if statement_type == "balance_sheet" or "balance sheet" in title or "financial position" in title:
            return item8.children["item_8_balance_sheet"]
        if statement_type == "cash_flow" or "cash flow" in title:
            return item8.children["item_8_cash_flows"]
        if "note " in title or "note " in text or statement_type in ("segment_table", "maturity_table", "note_disclosures"):
            note_match = re.search(r"note\s+(\d+)", title + " " + text)
            if note_match:
                note_num = note_match.group(1)
                note_id = f"item_8_note_{note_num}"
                full_title = title.strip() or f"Note {note_num}"
                if note_id not in item8.children["item_8_notes"].children:
                    item8.children["item_8_notes"].add_child(
                        DocTreeNode(note_id, full_title, level=4)
                    )
                else:
                    node = item8.children["item_8_notes"].children[note_id]
                    if len(full_title) > len(node.title):
                        node.title = full_title
                return item8.children["item_8_notes"].children[note_id]
            return item8.children["item_8_notes"]

        # Secondary checks for statement text
        if any(k in text for k in ("statements of operations", "statement of operations", "income statement")):
            return item8.children["item_8_operations"]
        if any(k in text for k in ("balance sheets", "balance sheet", "financial position")):
            return item8.children["item_8_balance_sheet"]
        if any(k in text for k in ("cash flows", "cash flow statement")):
            return item8.children["item_8_cash_flows"]

        return item8

    def prune(
        self,
        section_hint: str | None = None,
        statement_type: str | None = None,
    ) -> list[str]:
        """Prune document tree to retain only chunks relevant to section_hint or statement_type.

        Returns:
            list of candidate chunk_ids strictly within the pruned subtree.
            If section_hint does not match or is None, returns all chunks.
        """
        if not self._all_chunk_ids:
            return []

        if not section_hint and not statement_type:
            return sorted(self._all_chunk_ids)

        hint = (section_hint or "").lower()
        stype = (statement_type or "").lower()
        part2 = self.root.children["part_2"]
        item8 = part2.children["item_8"]

        # Case 1: MD&A in Item 7 (check first to avoid 'results of operations' in MD&A hitting statement)
        if "item 7" in hint or "md&a" in hint:
            return sorted(part2.children["item_7"].collect_chunk_ids())

        # Case 2: Specific Note or Note Topic
        if "note " in hint:
            note_match = re.search(r"note\s+(\d+)", hint)
            if note_match:
                note_num = note_match.group(1)
                note_id = f"item_8_note_{note_num}"
                if note_id in item8.children["item_8_notes"].children:
                    res = item8.children["item_8_notes"].children[note_id].collect_chunk_ids()
                    if res:
                        return sorted(res)
            # Match note topics dynamically across all notes in Item 8
            topic_terms = ("segment", "debt", "lease", "borrowing", "tax", "revenue", "commitment", "contingenc", "pension")
            for term in topic_terms:
                if term in hint:
                    matching_chunks: list[str] = []
                    for child in item8.children["item_8_notes"].children.values():
                        if term in child.title.lower():
                            matching_chunks.extend(child.collect_chunk_ids())
                    if matching_chunks:
                        return sorted(set(matching_chunks))
            return sorted(item8.children["item_8_notes"].collect_chunk_ids())

        # Case 2b: Note topic without explicit "note " string (e.g. "segment disclosures", "segment information")
        if "segment" in hint:
            matching_chunks: list[str] = []
            for child in item8.children["item_8_notes"].children.values():
                if "segment" in child.title.lower():
                    matching_chunks.extend(child.collect_chunk_ids())
            if matching_chunks:
                return sorted(set(matching_chunks))
            return sorted(item8.children["item_8_notes"].collect_chunk_ids())

        # Case 3: Specific financial statement
        if "operation" in hint or "income" in hint or stype == "income_statement":
            return sorted(item8.children["item_8_operations"].collect_chunk_ids())
        if "balance sheet" in hint or stype == "balance_sheet":
            return sorted(item8.children["item_8_balance_sheet"].collect_chunk_ids())
        if "cash flow" in hint or stype == "cash_flow":
            return sorted(item8.children["item_8_cash_flows"].collect_chunk_ids())

        # Case 4: Item 8 overall
        if "item 8" in hint or "financial statement" in hint:
            return sorted(item8.collect_chunk_ids())

        # Case 5: Item 1A Risk Factors
        if "item 1a" in hint or "risk" in hint:
            return sorted(self.root.children["part_1"].children["item_1a"].collect_chunk_ids())

        # Fallback to all chunks if hint was not resolved
        return sorted(self._all_chunk_ids)

    def pruning_ratio(self, pruned_chunks: list[str]) -> float:
        """Calculate the proportion of document chunks pruned away (0.0 to 1.0)."""
        if not self._all_chunk_ids:
            return 0.0
        return 1.0 - (len(pruned_chunks) / len(self._all_chunk_ids))
