"""
Alignment pipeline orchestrator.

Coordinates chunk loading, ASR emissions extraction, chunk/word alignment engine execution,
and result exporting via injected outbound adapters.
"""

import os
from typing import Any, List, Optional, Sequence, Tuple

from transcription.alignment.domain.models import AlignmentOutput
from transcription.alignment.ports.protocols import (
    ASREmissionsExtractor,
    ChunkAlignmentEngine,
    InboundChunkAdapter,
    OutboundAlignmentAdapter,
)


class AlignmentPipeline:
    """
    Orchestrator for the alignment lifecycle using dependency-injected ports.
    """

    def __init__(
        self,
        chunk_adapter: InboundChunkAdapter,
        extractor: ASREmissionsExtractor,
        engine: ChunkAlignmentEngine,
        exporters: Sequence[Tuple[OutboundAlignmentAdapter, str]],
    ):
        self.chunk_adapter = chunk_adapter
        self.extractor = extractor
        self.engine = engine
        self.exporters = list(exporters)

    def run(
        self,
        audio_input: Any,
        output_dir: Optional[str] = None,
    ) -> AlignmentOutput:
        """
        Executes the alignment pipeline:
        1. Loads ground-truth text chunks via chunk_adapter
        2. Extracts token emissions via extractor
        3. Executes alignment via engine
        4. Exports artifacts to output_dir via injected exporters
        """
        chunks = self.chunk_adapter.load_chunks()
        emissions = self.extractor.extract(audio_input)

        alignment = self.engine.align_chunks(
            emissions=emissions,
            chunks=chunks,
        )
        if isinstance(audio_input, str):
            alignment.source_id = audio_input

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for exporter, filename in self.exporters:
                custom_path = os.path.join(output_dir, filename)
                exporter.export(alignment, custom_path)

        return alignment
