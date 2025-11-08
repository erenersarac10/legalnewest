"""Provenance Recorder - Harvey/Legora CTO-Level Production-Grade
Records provenance and lineage of Turkish legal document parsing

Production Features:
- Data provenance tracking
- Lineage recording (source → processing → output)
- Audit trail for all operations
- Processing metadata (parser version, timestamp, user)
- Data quality provenance
- Reproducibility support
- Compliance and governance
- Provenance export for audit
- Statistics tracking
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from collections import defaultdict
import json
import hashlib

logger = logging.getLogger(__name__)


class ProvenanceType(Enum):
    """Types of provenance records"""
    PARSING = "PARSING"  # Document parsing
    TRANSFORMATION = "TRANSFORMATION"  # Data transformation
    VALIDATION = "VALIDATION"  # Validation operation
    ENRICHMENT = "ENRICHMENT"  # Data enrichment
    EXPORT = "EXPORT"  # Data export
    IMPORT = "IMPORT"  # Data import
    SNAPSHOT = "SNAPSHOT"  # Snapshot creation
    VERSION = "VERSION"  # Version creation
    CUSTOM = "CUSTOM"  # Custom operation


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "EXCELLENT"  # 95-100% confidence
    GOOD = "GOOD"  # 80-95% confidence
    ACCEPTABLE = "ACCEPTABLE"  # 60-80% confidence
    POOR = "POOR"  # 40-60% confidence
    UNACCEPTABLE = "UNACCEPTABLE"  # <40% confidence


class OperationStatus(Enum):
    """Operation status"""
    SUCCESS = "SUCCESS"  # Completed successfully
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"  # Partially successful
    FAILED = "FAILED"  # Failed
    PENDING = "PENDING"  # In progress
    CANCELLED = "CANCELLED"  # Cancelled


@dataclass
class Agent:
    """Represents an agent (user, system, or process)"""
    agent_id: str
    agent_type: str  # "user", "system", "parser", "validator"
    name: str
    version: Optional[str] = None  # For software agents
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.version:
            return f"{self.name} v{self.version}"
        return self.name


@dataclass
class DataEntity:
    """Represents a data entity in lineage"""
    entity_id: str
    entity_type: str  # "document", "article", "metadata"
    checksum: Optional[str] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceRecord:
    """Complete provenance record for an operation"""
    record_id: str
    provenance_type: ProvenanceType

    # Timing
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0

    # Agent (who/what performed operation)
    agent: Optional[Agent] = None

    # Data lineage
    input_entities: List[DataEntity] = field(default_factory=list)
    output_entities: List[DataEntity] = field(default_factory=list)

    # Operation details
    operation_name: str = ""
    operation_parameters: Dict[str, Any] = field(default_factory=dict)
    status: OperationStatus = OperationStatus.PENDING

    # Quality metrics
    data_quality: Optional[DataQuality] = None
    confidence_score: float = 0.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    # Turkish legal document specific
    document_type: Optional[str] = None  # law, regulation, decision
    source_reference: Optional[str] = None  # Original source reference

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Related records
    parent_record_id: Optional[str] = None
    child_record_ids: List[str] = field(default_factory=list)

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, entity: DataEntity) -> None:
        """Add input entity"""
        self.input_entities.append(entity)

    def add_output(self, entity: DataEntity) -> None:
        """Add output entity"""
        self.output_entities.append(entity)

    def add_error(self, error: str) -> None:
        """Add error"""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add warning"""
        self.warnings.append(warning)

    def complete(self, status: OperationStatus = OperationStatus.SUCCESS) -> None:
        """Mark operation as complete"""
        self.completed_at = datetime.now().isoformat()
        self.status = status

        # Calculate duration
        if self.started_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                end = datetime.fromisoformat(self.completed_at)
                self.duration_seconds = (end - start).total_seconds()
            except:
                pass

    def summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Provenance Record: {self.record_id}")
        lines.append(f"Type: {self.provenance_type.value}")
        lines.append(f"Operation: {self.operation_name}")
        lines.append(f"Status: {self.status.value}")

        if self.agent:
            lines.append(f"Agent: {self.agent}")

        lines.append(f"Duration: {self.duration_seconds:.2f}s")
        lines.append(f"Inputs: {len(self.input_entities)}")
        lines.append(f"Outputs: {len(self.output_entities)}")

        if self.data_quality:
            lines.append(f"Quality: {self.data_quality.value} ({self.confidence_score:.2f})")

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")

        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")

        return '\n'.join(lines)


@dataclass
class LineageGraph:
    """Lineage graph showing data flow"""
    nodes: Dict[str, DataEntity] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, via_record)
    records: Dict[str, ProvenanceRecord] = field(default_factory=dict)

    def add_node(self, entity: DataEntity) -> None:
        """Add entity node"""
        self.nodes[entity.entity_id] = entity

    def add_edge(self, from_id: str, to_id: str, via_record_id: str) -> None:
        """Add lineage edge"""
        self.edges.append((from_id, to_id, via_record_id))

    def add_record(self, record: ProvenanceRecord) -> None:
        """Add provenance record"""
        self.records[record.record_id] = record

        # Add entities as nodes
        for entity in record.input_entities:
            self.add_node(entity)

        for entity in record.output_entities:
            self.add_node(entity)

        # Add edges from inputs to outputs
        for input_entity in record.input_entities:
            for output_entity in record.output_entities:
                self.add_edge(
                    input_entity.entity_id,
                    output_entity.entity_id,
                    record.record_id
                )

    def get_ancestors(self, entity_id: str) -> Set[str]:
        """Get all ancestor entities"""
        ancestors = set()
        to_visit = {entity_id}

        while to_visit:
            current = to_visit.pop()
            for from_id, to_id, _ in self.edges:
                if to_id == current and from_id not in ancestors:
                    ancestors.add(from_id)
                    to_visit.add(from_id)

        return ancestors

    def get_descendants(self, entity_id: str) -> Set[str]:
        """Get all descendant entities"""
        descendants = set()
        to_visit = {entity_id}

        while to_visit:
            current = to_visit.pop()
            for from_id, to_id, _ in self.edges:
                if from_id == current and to_id not in descendants:
                    descendants.add(to_id)
                    to_visit.add(to_id)

        return descendants

    def export_graphviz(self) -> str:
        """Export as Graphviz DOT format"""
        lines = ['digraph lineage {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')

        # Add nodes
        for entity_id, entity in self.nodes.items():
            label = f"{entity.entity_type}\\n{entity_id[:8]}"
            lines.append(f'  "{entity_id}" [label="{label}"];')

        # Add edges
        for from_id, to_id, record_id in self.edges:
            lines.append(f'  "{from_id}" -> "{to_id}";')

        lines.append('}')
        return '\n'.join(lines)


class ProvenanceRecorder:
    """Provenance Recorder for Turkish Legal Document Parsing

    Records and tracks data provenance:
    - Operation tracking
    - Data lineage
    - Audit trail
    - Quality tracking
    - Reproducibility support
    - Compliance reporting

    Features:
    - Complete lineage graphs
    - Agent tracking
    - Turkish legal document specific
    - Export for audit
    - Statistics tracking
    """

    def __init__(self):
        """Initialize Provenance Recorder"""
        # Provenance records
        self.records: Dict[str, ProvenanceRecord] = {}

        # Lineage graph
        self.lineage = LineageGraph()

        # Agent registry
        self.agents: Dict[str, Agent] = {}

        # Entity registry
        self.entities: Dict[str, DataEntity] = {}

        # Statistics
        self.stats = {
            'total_records': 0,
            'total_operations': defaultdict(int),
            'operation_status': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'total_errors': 0,
            'total_warnings': 0,
        }

        logger.info("Initialized Provenance Recorder")

    def start_operation(
        self,
        provenance_type: ProvenanceType,
        operation_name: str,
        agent: Optional[Agent] = None,
        **kwargs
    ) -> ProvenanceRecord:
        """Start recording an operation

        Args:
            provenance_type: Type of provenance
            operation_name: Operation name
            agent: Agent performing operation
            **kwargs: Additional parameters

        Returns:
            ProvenanceRecord for tracking
        """
        # Generate record ID
        record_id = self._generate_record_id(provenance_type)

        # Create record
        record = ProvenanceRecord(
            record_id=record_id,
            provenance_type=provenance_type,
            started_at=datetime.now().isoformat(),
            agent=agent,
            operation_name=operation_name,
            operation_parameters=kwargs.get('parameters', {}),
            document_type=kwargs.get('document_type'),
            source_reference=kwargs.get('source_reference'),
            parent_record_id=kwargs.get('parent_record_id'),
            tags=kwargs.get('tags', []),
            metadata=kwargs.get('metadata', {})
        )

        # Store record
        self.records[record_id] = record

        logger.info(f"Started operation {record_id}: {operation_name}")
        return record

    def complete_operation(
        self,
        record: ProvenanceRecord,
        status: OperationStatus = OperationStatus.SUCCESS,
        **kwargs
    ) -> None:
        """Complete an operation

        Args:
            record: Provenance record to complete
            status: Operation status
            **kwargs: Additional data
                - data_quality: Data quality level
                - confidence_score: Confidence score
                - quality_metrics: Quality metrics
        """
        # Complete record
        record.complete(status)

        # Update quality
        if 'data_quality' in kwargs:
            record.data_quality = kwargs['data_quality']

        if 'confidence_score' in kwargs:
            record.confidence_score = kwargs['confidence_score']

        if 'quality_metrics' in kwargs:
            record.quality_metrics = kwargs['quality_metrics']

        # Update lineage graph
        self.lineage.add_record(record)

        # Update parent-child relationships
        if record.parent_record_id and record.parent_record_id in self.records:
            parent = self.records[record.parent_record_id]
            parent.child_record_ids.append(record.record_id)

        # Update statistics
        self._update_stats(record)

        logger.info(f"Completed operation {record.record_id}: {status.value}")

    def record_parsing(
        self,
        parser_name: str,
        parser_version: str,
        input_data: Any,
        output_data: Any,
        **kwargs
    ) -> ProvenanceRecord:
        """Record a parsing operation

        Args:
            parser_name: Parser name
            parser_version: Parser version
            input_data: Input data
            output_data: Output data
            **kwargs: Additional options

        Returns:
            ProvenanceRecord
        """
        # Create agent
        agent = Agent(
            agent_id=f"parser_{parser_name}",
            agent_type="parser",
            name=parser_name,
            version=parser_version
        )

        self.agents[agent.agent_id] = agent

        # Start operation
        record = self.start_operation(
            provenance_type=ProvenanceType.PARSING,
            operation_name=f"Parse with {parser_name}",
            agent=agent,
            **kwargs
        )

        # Create entities
        input_entity = self._create_entity(input_data, "input_document")
        output_entity = self._create_entity(output_data, "parsed_document")

        record.add_input(input_entity)
        record.add_output(output_entity)

        # Complete operation
        status = kwargs.get('status', OperationStatus.SUCCESS)
        self.complete_operation(record, status, **kwargs)

        return record

    def record_validation(
        self,
        validator_name: str,
        data: Any,
        validation_result: Any,
        **kwargs
    ) -> ProvenanceRecord:
        """Record a validation operation

        Args:
            validator_name: Validator name
            data: Data being validated
            validation_result: Validation result
            **kwargs: Additional options

        Returns:
            ProvenanceRecord
        """
        # Create agent
        agent = Agent(
            agent_id=f"validator_{validator_name}",
            agent_type="validator",
            name=validator_name
        )

        # Start operation
        record = self.start_operation(
            provenance_type=ProvenanceType.VALIDATION,
            operation_name=f"Validate with {validator_name}",
            agent=agent,
            **kwargs
        )

        # Create entities
        data_entity = self._create_entity(data, "validated_data")
        result_entity = self._create_entity(validation_result, "validation_result")

        record.add_input(data_entity)
        record.add_output(result_entity)

        # Add validation errors/warnings
        if hasattr(validation_result, 'errors'):
            for error in validation_result.errors:
                record.add_error(str(error))

        if hasattr(validation_result, 'warnings'):
            for warning in validation_result.warnings:
                record.add_warning(str(warning))

        # Complete operation
        status = OperationStatus.SUCCESS if not record.errors else OperationStatus.PARTIAL_SUCCESS
        self.complete_operation(record, status, **kwargs)

        return record

    def record_transformation(
        self,
        transformation_name: str,
        input_data: Any,
        output_data: Any,
        **kwargs
    ) -> ProvenanceRecord:
        """Record a data transformation

        Args:
            transformation_name: Transformation name
            input_data: Input data
            output_data: Output data
            **kwargs: Additional options

        Returns:
            ProvenanceRecord
        """
        # Start operation
        record = self.start_operation(
            provenance_type=ProvenanceType.TRANSFORMATION,
            operation_name=transformation_name,
            **kwargs
        )

        # Create entities
        input_entity = self._create_entity(input_data, "input")
        output_entity = self._create_entity(output_data, "output")

        record.add_input(input_entity)
        record.add_output(output_entity)

        # Complete operation
        status = kwargs.get('status', OperationStatus.SUCCESS)
        self.complete_operation(record, status, **kwargs)

        return record

    def record_enrichment(
        self,
        enrichment_name: str,
        base_data: Any,
        enriched_data: Any,
        enrichment_source: str,
        **kwargs
    ) -> ProvenanceRecord:
        """Record data enrichment

        Args:
            enrichment_name: Enrichment operation name
            base_data: Base data
            enriched_data: Enriched data
            enrichment_source: Source of enrichment
            **kwargs: Additional options

        Returns:
            ProvenanceRecord
        """
        # Start operation
        record = self.start_operation(
            provenance_type=ProvenanceType.ENRICHMENT,
            operation_name=enrichment_name,
            parameters={'enrichment_source': enrichment_source},
            **kwargs
        )

        # Create entities
        base_entity = self._create_entity(base_data, "base_data")
        enriched_entity = self._create_entity(enriched_data, "enriched_data")

        record.add_input(base_entity)
        record.add_output(enriched_entity)

        # Complete operation
        status = kwargs.get('status', OperationStatus.SUCCESS)
        self.complete_operation(record, status, **kwargs)

        return record

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get provenance record by ID

        Args:
            record_id: Record ID

        Returns:
            ProvenanceRecord or None
        """
        return self.records.get(record_id)

    def get_records_by_type(self, provenance_type: ProvenanceType) -> List[ProvenanceRecord]:
        """Get records by type

        Args:
            provenance_type: Provenance type

        Returns:
            List of records
        """
        return [r for r in self.records.values() if r.provenance_type == provenance_type]

    def get_lineage(self, entity_id: str) -> Dict[str, Any]:
        """Get complete lineage for an entity

        Args:
            entity_id: Entity ID

        Returns:
            Lineage information
        """
        ancestors = self.lineage.get_ancestors(entity_id)
        descendants = self.lineage.get_descendants(entity_id)

        # Get all related records
        related_records = set()
        for from_id, to_id, record_id in self.lineage.edges:
            if from_id == entity_id or to_id == entity_id:
                related_records.add(record_id)
            if from_id in ancestors or to_id in descendants:
                related_records.add(record_id)

        return {
            'entity_id': entity_id,
            'ancestors': list(ancestors),
            'descendants': list(descendants),
            'related_records': list(related_records),
            'lineage_depth': len(ancestors) + len(descendants)
        }

    def export_audit_trail(
        self,
        entity_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export audit trail

        Args:
            entity_id: Filter by entity
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Audit trail data
        """
        # Filter records
        records = list(self.records.values())

        if entity_id:
            # Get records related to entity
            lineage = self.get_lineage(entity_id)
            related_record_ids = set(lineage['related_records'])
            records = [r for r in records if r.record_id in related_record_ids]

        if start_date:
            records = [r for r in records if r.started_at >= start_date]

        if end_date:
            records = [r for r in records if r.started_at <= end_date]

        # Build audit trail
        audit_trail = {
            'generated_at': datetime.now().isoformat(),
            'filter': {
                'entity_id': entity_id,
                'start_date': start_date,
                'end_date': end_date
            },
            'summary': {
                'total_records': len(records),
                'operation_types': defaultdict(int),
                'status_distribution': defaultdict(int),
                'total_errors': 0,
                'total_warnings': 0
            },
            'records': []
        }

        # Process records
        for record in sorted(records, key=lambda r: r.started_at):
            # Update summary
            audit_trail['summary']['operation_types'][record.provenance_type.value] += 1
            audit_trail['summary']['status_distribution'][record.status.value] += 1
            audit_trail['summary']['total_errors'] += len(record.errors)
            audit_trail['summary']['total_warnings'] += len(record.warnings)

            # Add record
            audit_trail['records'].append({
                'record_id': record.record_id,
                'type': record.provenance_type.value,
                'operation': record.operation_name,
                'started_at': record.started_at,
                'completed_at': record.completed_at,
                'duration_seconds': record.duration_seconds,
                'status': record.status.value,
                'agent': str(record.agent) if record.agent else None,
                'inputs': [e.entity_id for e in record.input_entities],
                'outputs': [e.entity_id for e in record.output_entities],
                'errors': record.errors,
                'warnings': record.warnings,
                'quality': record.data_quality.value if record.data_quality else None,
                'confidence': record.confidence_score
            })

        logger.info(f"Exported audit trail with {len(records)} records")
        return audit_trail

    def export_lineage_graph(self, format: str = 'json') -> Any:
        """Export lineage graph

        Args:
            format: Export format ('json', 'graphviz')

        Returns:
            Exported graph data
        """
        if format == 'json':
            return {
                'nodes': {
                    entity_id: {
                        'entity_id': entity.entity_id,
                        'entity_type': entity.entity_type,
                        'checksum': entity.checksum,
                        'size_bytes': entity.size_bytes
                    }
                    for entity_id, entity in self.lineage.nodes.items()
                },
                'edges': [
                    {
                        'from': from_id,
                        'to': to_id,
                        'via_record': record_id
                    }
                    for from_id, to_id, record_id in self.lineage.edges
                ]
            }
        elif format == 'graphviz':
            return self.lineage.export_graphviz()
        else:
            logger.error(f"Unsupported export format: {format}")
            return None

    def verify_reproducibility(self, record_id: str) -> Dict[str, Any]:
        """Verify if operation is reproducible

        Args:
            record_id: Record ID to verify

        Returns:
            Reproducibility report
        """
        record = self.get_record(record_id)
        if not record:
            return {'reproducible': False, 'reason': 'Record not found'}

        report = {
            'reproducible': True,
            'record_id': record_id,
            'issues': []
        }

        # Check if all inputs have checksums
        for entity in record.input_entities:
            if not entity.checksum:
                report['reproducible'] = False
                report['issues'].append(f"Input entity {entity.entity_id} has no checksum")

        # Check if operation parameters are recorded
        if not record.operation_parameters:
            report['issues'].append("No operation parameters recorded")

        # Check if agent version is recorded
        if record.agent and not record.agent.version:
            report['issues'].append("Agent version not recorded")

        # Check operation status
        if record.status != OperationStatus.SUCCESS:
            report['reproducible'] = False
            report['issues'].append(f"Operation status: {record.status.value}")

        return report

    def _create_entity(self, data: Any, entity_type: str) -> DataEntity:
        """Create data entity from data

        Args:
            data: Data to create entity from
            entity_type: Entity type

        Returns:
            DataEntity
        """
        # Calculate checksum
        checksum = self._calculate_checksum(data)

        # Generate entity ID
        entity_id = f"{entity_type}_{checksum[:12]}"

        # Calculate size
        size_bytes = self._calculate_size(data)

        # Create entity
        entity = DataEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            checksum=checksum,
            size_bytes=size_bytes
        )

        # Store entity
        self.entities[entity_id] = entity

        return entity

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum of data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes"""
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, ensure_ascii=False)
            return len(json_str.encode('utf-8'))
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        else:
            return len(str(data).encode('utf-8'))

    def _generate_record_id(self, provenance_type: ProvenanceType) -> str:
        """Generate unique record ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        type_code = provenance_type.value[:4]
        return f"PROV_{type_code}_{timestamp}"

    def _update_stats(self, record: ProvenanceRecord) -> None:
        """Update statistics"""
        self.stats['total_records'] += 1
        self.stats['total_operations'][record.provenance_type.value] += 1
        self.stats['operation_status'][record.status.value] += 1

        if record.data_quality:
            self.stats['quality_distribution'][record.data_quality.value] += 1

        self.stats['total_processing_time'] += record.duration_seconds
        self.stats['total_errors'] += len(record.errors)
        self.stats['total_warnings'] += len(record.warnings)

        # Update average
        if self.stats['total_records'] > 0:
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_records']
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get recorder statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'total_records': 0,
            'total_operations': defaultdict(int),
            'operation_status': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'total_errors': 0,
            'total_warnings': 0,
        }
        logger.info("Statistics reset")


__all__ = [
    'ProvenanceRecorder',
    'ProvenanceRecord',
    'LineageGraph',
    'Agent',
    'DataEntity',
    'ProvenanceType',
    'DataQuality',
    'OperationStatus'
]
