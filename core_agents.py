# -*- coding: utf-8 -*-
from __future__ import annotations
"""Data_Architect_Agent_05Oct2025.ipynb

"""

# ============================================================================
# DATA ARCHITECT AGENT - MAIN IMPLEMENTATION
# ============================================================================

# Required installs


# -*- coding: utf-8 -*-
"""
Data Architect Agent - Clean Version
This notebook implements a comprehensive Data Architect Agent system
"""

# Required installs

# Required imports
import json
import logging
import re
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import random
import string
import sqlite3
import traceback
from glob import glob

import pandas as pd
import numpy as np
from faker import Faker

import mysql.connector
from mysql.connector import Error

# === CONFIGURATION: Suppress non-critical warnings ===
import warnings
import logging

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*infer_datetime_format.*')
warnings.filterwarnings('ignore', message='.*Could not infer format.*')

# Reduce logging verbosity for data inference
logging.getLogger('DataInferenceAgent').setLevel(logging.ERROR)

print("✅ Warning suppression configured - UI will be cleaner")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Open AI API Key
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Agent Temperatures - settable per agent
AGENT_TEMPERATURES = {
    "requirements": 0.8,
    "conceptual": 0.3,
    "logical": 0.2,
    "physical": 0.1,
    "glossary": 0.7,
    "data_product": 0.3,
    "ontology": 0.4
}

@dataclass
class ProjectConfig:
    """Configuration for project structure"""
    project_id: str
    base_path: str = "./projects"
    create_zip: bool = True
    include_readme: bool = True

    def get_project_path(self) -> Path:
        return Path(self.base_path) / self.project_id

class TemperatureAdjustedLLM:
    """Wrapper to allow different temperatures for different agents"""

    def __init__(self, base_llm, agent_temperatures=None):
        """
        Initialize with a base LLM and temperature map

        Args:
            base_llm: The base LLM instance
            agent_temperatures: Dict mapping agent types to temperatures
        """
        self.base_llm = base_llm
        self.agent_temperatures = agent_temperatures or AGENT_TEMPERATURES
        self.llm_cache = {}

    def get_llm_for_agent(self, agent_type):
        """Get an LLM with the appropriate temperature for this agent type"""
        if agent_type not in self.agent_temperatures:
            return self.base_llm

        # Check if we've already created this LLM
        if agent_type in self.llm_cache:
            return self.llm_cache[agent_type]

        # Create a new LLM with the right temperature
        from langchain_openai import ChatOpenAI
        try:
            # Get model name from base LLM if possible
            model_name = self.base_llm.model_name if hasattr(self.base_llm, "model_name") else "gpt-4"
            temperature = self.agent_temperatures[agent_type]

            # Create new LLM with adjusted temperature
            adjusted_llm = ChatOpenAI(model=model_name, temperature=temperature)
            self.llm_cache[agent_type] = adjusted_llm
            return adjusted_llm
        except Exception as e:
            print(f"Warning: Couldn't create LLM for {agent_type}: {e}")
            return self.base_llm

    def invoke(self, *args, **kwargs):
        """Default invoke just passes through to base LLM"""
        return self.base_llm.invoke(*args, **kwargs)

@dataclass
class SimpleTableInfo:
    """Simple table information structure"""
    name: str
    columns: List[Dict[str, Any]]

class SimpleDDLParser:
    """Simple parser that extracts table information from DDL"""

    def parse_ddl(self, ddl_content: str) -> List[SimpleTableInfo]:
        """Parse DDL and return table information"""
        tables = []

        # Find all CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        matches = re.findall(create_table_pattern, ddl_content, re.DOTALL | re.IGNORECASE)

        for table_name, columns_text in matches:
            columns = self._parse_columns(columns_text)
            tables.append(SimpleTableInfo(name=table_name, columns=columns))

        return tables

    def _parse_columns(self, columns_text: str) -> List[Dict[str, Any]]:
        """Parse column definitions from CREATE TABLE statement"""
        columns = []

        # Split by comma, but be careful about nested parentheses
        column_lines = []
        current_line = ""
        paren_count = 0

        for char in columns_text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                column_lines.append(current_line.strip())
                current_line = ""
                continue
            current_line += char

        if current_line.strip():
            column_lines.append(current_line.strip())

        # Parse each column line
        for line in column_lines:
            line = line.strip()
            if not line or line.upper().startswith(('CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK')):
                continue

            # Basic column parsing: name type [constraints...]
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0]
                col_type = parts[1]

                # Check for constraints
                is_primary = 'PRIMARY KEY' in line.upper()
                is_not_null = 'NOT NULL' in line.upper()

                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'is_primary_key': is_primary,
                    'nullable': not is_not_null
                })

        return columns

class SimpleArtifactManager:
    """Manages creation and organization of project artifacts"""

    def __init__(self, project_config: ProjectConfig):
        self.config = project_config
        self.project_path = project_config.get_project_path()
        self.created_files = []
        self._setup_directories()

    def _setup_directories(self):
        """Create basic project directories"""
        directories = [
            "00_data_inference",
            "01_requirements",
            "02_models",
            "03_implementation",
            "04_testing",
            "05_data_products",
            "06_ontology",
            "07_documentation"
        ]

        self.project_path.mkdir(parents=True, exist_ok=True)

        for dir_name in directories:
            dir_path = self.project_path / dir_name
            dir_path.mkdir(exist_ok=True)

            # Create gitkeep with description
            gitkeep = dir_path / ".gitkeep"
            with open(gitkeep, 'w', encoding='utf-8') as f:
                f.write(f"# {dir_name}\n")

    def get_inference_dir(self) -> Path:
        p = self.project_path / "00_data_inference"
        p.mkdir(exist_ok=True, parents=True)
        return p

    def register_created_files(self, paths: list):
        for p in paths:
            self.created_files.append(str(p))

    def save_requirements(self, requirements: Dict[str, Any]) -> str:
        """Save requirements to JSON file"""
        req_dir = self.project_path / "01_requirements"

        # Save JSON
        json_file = req_dir / "requirements.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(requirements, f, indent=2)
        self.created_files.append(str(json_file))

        # Save Markdown summary
        md_file = req_dir / "requirements_summary.md"
        md_content = self._generate_requirements_markdown(requirements)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        self.created_files.append(str(md_file))

        return str(json_file)

    def save_model(self, model_content: str, model_type: str) -> str:
        """Save model to XMI file"""
        models_dir = self.project_path / "02_models"
        xmi_file = models_dir / (model_type + "_model.xmi")

        with open(xmi_file, 'w', encoding='utf-8') as f:
            f.write(model_content)
        self.created_files.append(str(xmi_file))

        return str(xmi_file)

    def save_dbml(self, dbml_content: str, model_type: str) -> str:
        """Save model to DBML file"""
        models_dir = self.project_path / "02_models"
        dbml_file = models_dir / (model_type + "_model.dbml")

        with open(dbml_file, 'w', encoding='utf-8') as f:
            f.write(dbml_content)
        self.created_files.append(str(dbml_file))

        return str(dbml_file)

    def save_implementation_artifacts(self, artifacts: Dict[str, Any]) -> List[str]:
        """Save implementation artifacts"""
        impl_dir = self.project_path / "03_implementation"
        saved_files = []

        if artifacts.get("sql_ddl"):
            sql_file = impl_dir / "database_schema.sql"
            with open(sql_file, 'w', encoding='utf-8') as f:
                f.write(artifacts["sql_ddl"])
            saved_files.append(str(sql_file))

        self.created_files.extend(saved_files)
        return saved_files

    def save_glossary(self, glossary: Dict[str, Any]) -> str:
        """Save glossary to JSON and Markdown files"""
        doc_dir = self.project_path / "07_documentation"

        # Save JSON
        json_file = doc_dir / "glossary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(glossary, f, indent=2)
        self.created_files.append(str(json_file))

        # Save Markdown
        md_file = doc_dir / "glossary.md"
        md_content = self._generate_glossary_markdown(glossary)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        self.created_files.append(str(md_file))

        return str(md_file)

    def save_data_product_config(self, config: Dict[str, Any]) -> List[str]:
        """Save data product configurations"""
        dp_dir = self.project_path / "05_data_products"
        saved_files = []

        if config.get("api_specification"):
            api_file = dp_dir / "api_specification.json"
            with open(api_file, 'w', encoding='utf-8') as f:
                json.dump(config["api_specification"], f, indent=2)
            saved_files.append(str(api_file))

        self.created_files.extend(saved_files)
        return saved_files

    def save_ontology_artifacts(self, artifacts: Dict[str, Any]) -> List[str]:
        """Save ontology artifacts to OWL and JSON-LD files"""
        ontology_dir = self.project_path / "06_ontology"
        saved_files = []

        if artifacts.get("owl_ontology"):
            owl_file = ontology_dir / "domain_ontology.owl"
            with open(owl_file, 'w', encoding='utf-8') as f:
                f.write(artifacts["owl_ontology"])
            saved_files.append(str(owl_file))

        if artifacts.get("jsonld_context"):
            jsonld_file = ontology_dir / "jsonld_context.json"
            with open(jsonld_file, 'w', encoding='utf-8') as f:
                json.dump(artifacts["jsonld_context"], f, indent=2)
            saved_files.append(str(jsonld_file))

        self.created_files.extend(saved_files)
        return saved_files

    def generate_readme(self) -> str:
        """Generate project README"""
        readme_file = self.project_path / "README.md"

        readme_content = f"""# Data Architecture Project: {self.config.project_id}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Generated by:** Data Architect Agent System

## Project Structure
{self.config.project_id}/
├── 00_data_inference/        # Data Inferencing
├── 01_requirements/          # Requirements analysis
├── 02_models/               # Data models (conceptual, logical, physical)
├── 03_implementation/       # SQL DDL and scripts
├── 04_testing/             # Test suites
├── 05_data_products/       # API specifications
├── 06_ontology/            # Ontology artifacts
├── 07_documentation/       # Documentation
└── README.md              # This file
## Files Generated

{len(self.created_files)} files were automatically generated.

## Next Steps

1. Review requirements analysis
2. Examine data models
3. Execute SQL scripts
4. Deploy APIs
5. Run tests

---
*Generated by Data Architect Agent System*
"""

        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        self.created_files.append(str(readme_file))
        return str(readme_file)

    def _generate_requirements_markdown(self, requirements: Dict[str, Any]) -> str:
        """Generate Markdown summary of requirements"""
        content_parts = [
            f"# Requirements Analysis",
            f"",
            f"**Project:** {self.config.project_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Functional Requirements",
            self._format_list(requirements.get('functional_requirements', [])),
            f"",
            f"## Non-Functional Requirements",
            self._format_list(requirements.get('non_functional_requirements', [])),
            f"",
            f"## Data Requirements",
            self._format_list(requirements.get('data_requirements', [])),
            f"",
            f"## Integration Requirements",
            self._format_list(requirements.get('integration_requirements', [])),
            f"",
            f"## Compliance Requirements",
            self._format_list(requirements.get('compliance_requirements', []))
        ]

        return "\n".join(content_parts)

    def _generate_glossary_markdown(self, glossary: Dict[str, Any]) -> str:
        """Generate Markdown representation of glossary"""
        content_parts = [
            f"# {glossary.get('title', 'Domain Glossary')}",
            f"",
            f"**Project:** {self.config.project_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"{glossary.get('description', 'Glossary of domain-specific terms and definitions')}",
            f"",
            f"## Terms and Definitions",
            f""
        ]

        entries = glossary.get('entries', [])
        for entry in entries:
            content_parts.append(f"### {entry.get('term')}")
            content_parts.append(f"{entry.get('definition')}")
            content_parts.append(f"")

        return "\n".join(content_parts)

    def _format_list(self, items: List[str]) -> str:
        """Format list items as Markdown"""
        if not items:
            return "- None specified"
        return '\n'.join(f"- {item}" for item in items)

# ================================================================
# DataInferenceAgent (Notebook Version)
# ================================================================
# Usage inside orchestrator:
#   result = self.agents["data_inference"](csv_files, db_url=None)
# ================================================================

import re, sys, logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Float, String, Boolean,
    Date, DateTime, BigInteger, Text, ForeignKey, inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable


# -----------------------------
# Utility helpers
# -----------------------------

def _normalize_entity_name(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base)
    base = re.sub(r"__+", "_", base).strip("_").lower()
    return base

def _to_snake(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_").lower()
    return s

def _infer_series_type(s: pd.Series) -> str:
    non_null = s.dropna()
    if non_null.empty:
        return "string"
    # boolean
    if set(non_null.unique()).issubset(
        {True, False, 1, 0, "1", "0", "true", "false", "True", "False"}
    ):
        return "boolean"
    # integer / bigint
    try:
        as_int = pd.to_numeric(non_null, errors="coerce", downcast="integer")
        if as_int.notna().mean() > 0.98:
            if abs(int(as_int.max())) > 2**31 - 1:
                return "bigint"
            return "integer"
    except Exception: pass
    # float
    try:
        as_float = pd.to_numeric(non_null, errors="coerce")
        if as_float.notna().mean() > 0.98:
            return "float"
    except Exception: pass
    # datetime
    try:
        as_dt = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
        if as_dt.notna().mean() > 0.95:
            if any(getattr(x, "hour", 0) or getattr(x, "minute", 0) for x in as_dt.dropna()[:100]):
                return "datetime"
            return "date"
    except Exception: pass
    # text vs string
    if non_null.astype(str).map(len).max() > 500:
        return "text"
    return "string"

def _sqlalchemy_type(t: str):
    return {
        "bigint": BigInteger,
        "integer": Integer,
        "float": Float,
        "boolean": Boolean,
        "date": Date,
        "datetime": DateTime,
        "text": Text,
    }.get(t, String)


# -----------------------------
# Schema dataclasses
# -----------------------------

@dataclass
class Attribute:
    name: str
    type: str
    nullable: bool = True
    example: Optional[str] = None

@dataclass
class Entity:
    name: str
    attributes: List[Attribute] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    from_entity: str
    from_columns: List[str]
    to_entity: str
    to_columns: List[str]
    cardinality: str = "many-to-one"


# -----------------------------
# Main Agent
# -----------------------------

class DataInferenceAgent:
    def __init__(self,
        output_dir: str | Path = "./inference_artifacts",
        db_url: Optional[str] = None,
        prefer_mysql: bool = False,
        echo_sql: bool = False,
        fk_coverage_threshold: float = 0.80,
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_url = db_url
        self.prefer_mysql = prefer_mysql
        self.echo_sql = echo_sql
        self.fk_coverage_threshold = fk_coverage_threshold
        self.logger = logger or self._default_logger()

    def _default_logger(self):
        logger = logging.getLogger("DataInferenceAgent")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(ch)
        return logger

    # ---- Public API ----
    # def run(self, files: List[str]) -> Dict[str, Any]:
    #     schema = self.infer_schema(files)
    #     artifacts = self.generate_artifacts(schema)
    #     db_result = self.provision_and_load(schema, files)
    #     return {"schema": schema, "artifacts": artifacts, "database": db_result}

    # def run(self, files: List[str]) -> Dict[str, Any]:
    #   schema = self.infer_schema(files)
    #   artifacts = self.generate_artifacts(schema)
    #   db_result = self.provision_and_load(schema, files)

    #   # Extract entity names for downstream agents
    #   entity_names = []
    #   for entity_info in schema.get("entities", []):
    #       name = entity_info.get("name", "")
    #       # Convert to singular PascalCase for entity name
    #       entity_name = self._table_to_entity_name(name)
    #       entity_names.append(entity_name)

    #   return {
    #       "schema": schema,
    #       "artifacts": artifacts,
    #       "database": db_result,
    #       "entity_names": entity_names  # Add this
    #   }

    # def _table_to_entity_name(self, table_name: str) -> str:
    #     """Convert table name (plural, lowercase) to entity name (singular, PascalCase)"""
    #     # Make singular
    #     name = table_name.lower()
    #     if name.endswith('ies'):
    #         name = name[:-3] + 'y'
    #     elif name.endswith('es'):
    #         name = name[:-2]
    #     elif name.endswith('s') and not name.endswith('ss'):
    #         name = name[:-1]

    #     # Convert to PascalCase
    #     return ''.join(word.capitalize() for word in name.split('_'))

    def run(self, files: List[str]) -> Dict[str, Any]:
        schema = self.infer_schema(files)
        artifacts = self.generate_artifacts(schema)
        db_result = self.provision_and_load(schema, files)

        # Extract proper entity names from the schema
        entity_names = []
        for entity_dict in schema.get("entities", []):
            table_name = entity_dict.get("name", "")
            # Convert plural table name to singular entity name
            entity_name = self._table_to_entity_name(table_name)
            entity_names.append(entity_name)

        print(f"   DataInferenceAgent: Extracted entities: {entity_names}")

        return {
            "schema": schema,
            "artifacts": artifacts,
            "database": db_result,
            "entity_names": entity_names
        }

    def _table_to_entity_name(self, table_name: str) -> str:
        """Convert table name to proper entity name (singular, PascalCase)"""
        name = table_name.lower()

        # Handle common pluralizations
        if name.endswith('ies'):
            name = name[:-3] + 'y'  # companies -> company
        elif name.endswith('ses') or name.endswith('xes') or name.endswith('zes'):
            name = name[:-2]  # databases -> database
        elif name.endswith('s') and not name.endswith('ss'):
            name = name[:-1]  # customers -> customer

        # Convert to PascalCase
        return ''.join(word.capitalize() for word in name.split('_'))

    # ---- Step 1: Schema inference ----
    def infer_schema(self, files: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Inferencing schema from {len(files)} CSV file(s).")
        entities, dataframes = {}, {}
        for f in files:
            name = _normalize_entity_name(Path(f).stem)
            df = pd.read_csv(f)
            df.columns = [_to_snake(c) for c in df.columns]
            dataframes[name] = df
            attrs = []
            for col in df.columns:
                t = _infer_series_type(df[col])
                # attrs.append(Attribute(col, t, df[col].isna().any(),
                #                        str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""))
                attrs.append(Attribute(col, t, bool(df[col].isna().any()),
                       str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""))
            entities[name] = Entity(name, attrs)
        # PK inference (simplified: use 'id' or first unique col)
        for name, ent in entities.items():
            df = dataframes[name]
            if "id" in df.columns and df["id"].is_unique:
                ent.primary_key = ["id"]
            else:
                for c in df.columns:
                    if df[c].is_unique:
                        ent.primary_key = [c]; break
                if not ent.primary_key:
                    df["id"] = pd.RangeIndex(1, len(df)+1)
                    ent.attributes.insert(0, Attribute("id","integer",False,"1"))
                    ent.primary_key = ["id"]
                    dataframes[name] = df
        # FK inference (simple name matching + value overlap)
        relationships = []
        for src, df in dataframes.items():
            for col in df.columns:
                if col == "id" or col in entities[src].primary_key: continue
                if col.endswith("_id"):
                    tgt = col[:-3]
                    if tgt in entities and "id" in entities[tgt].primary_key:
                        coverage = len(set(df[col]) & set(dataframes[tgt]["id"])) / max(1,len(set(df[col])))
                        if coverage >= self.fk_coverage_threshold:
                            relationships.append(Relationship(src,[col],tgt,["id"]))
                            self.logger.info(f"[FK] {src}.{col} -> {tgt}.id")
        return {"entities":[{"name":e.name,"attributes":[a.__dict__ for a in e.attributes],
                             "primary_key":e.primary_key} for e in entities.values()],
                "relationships":[r.__dict__ for r in relationships]}

    # ---- Step 2: Artifact generation ----
    def generate_artifacts(self, schema: Dict[str, Any]) -> Dict[str, str]:
        out = {}
        # Mermaid ER
        lines = ["erDiagram"]
        for e in schema["entities"]:
            lines.append(f"  {e['name']} {{")
            for a in e["attributes"]:
                lines.append(f"    {a['type'].upper()} {a['name']}")
            lines.append("  }")
        for r in schema["relationships"]:
            lines.append(f"  {r['from_entity']} }}o--|| {r['to_entity']} : {','.join(r['from_columns'])}->{','.join(r['to_columns'])}")
        (self.output_dir/"conceptual_er_diagram.mmd").write_text("\n".join(lines))
        out["mermaid_er"] = str(self.output_dir/"conceptual_er_diagram.mmd")
        # JSON schema
        import json
        (self.output_dir/"logical_schema.json").write_text(json.dumps(schema,indent=2))
        out["logical_schema"] = str(self.output_dir/"logical_schema.json")
        # --- BEGIN: emit simple SQLite DDL from the inferred schema ---
        from sqlalchemy import MetaData, Table, Column
        from sqlalchemy.schema import CreateTable
        from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

        md = MetaData()
        for e in schema["entities"]:
            cols = []
            for a in e["attributes"]:
                cols.append(Column(
                    a["name"],
                    _sqlalchemy_type(a["type"])(),
                    primary_key=(a["name"] in e["primary_key"])
                ))
            Table(e["name"], md, *cols)

        ddl_lines = []
        for t in md.sorted_tables:
            ddl_lines.append(str(CreateTable(t).compile(dialect=sqlite_dialect())))

        ddl_path = self.output_dir / "physical_schema_sqlite.sql"
        ddl_path.write_text(";\n\n".join(ddl_lines) + ";\n")
        out["ddl_sqlite"] = str(ddl_path)
        # --- END: emit simple SQLite DDL ---

        return out

    # ---- Step 3: Provision + load ----
    def provision_and_load(self, schema: Dict[str, Any], files: List[str]) -> Dict[str, Any]:
        eng = self._get_engine()
        self._create_schema(eng, schema)
        report = self._load_data(eng, schema, files)

        # Get the actual database path
        db_path = None
        if hasattr(self, 'project_id'):
            db_path = str(Path(self.output_dir).parent / f"{self.project_id}_db.db")
        else:
            db_path = str(Path(self.output_dir).parent / "inferred_db.db")

        return {
            "db_url": str(eng.url),
            "load_report": report,
            "database_path": db_path  # Add this so we know where the database actually is
        }

    def _get_engine(self) -> Engine:
        if self.db_url:
            return create_engine(self.db_url, echo=self.echo_sql, future=True)

        # Get project name from the agent if set by orchestrator
        db_name = "inferred"
        if hasattr(self, 'project_id'):
            db_name = self.project_id

        # Place database in the project root directory, NOT in the inference subdirectory
        sqlite_path = Path(self.output_dir).parent / f"{db_name}_db.db"

        print(f"   Creating SQLite database: {sqlite_path}")

        # Remove old database if it exists
        if sqlite_path.exists():
            sqlite_path.unlink()

        eng = create_engine(f"sqlite:///{sqlite_path}", echo=self.echo_sql, future=True)
        with eng.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = ON;")
        return eng

        # Place database in the project root, not in subdirectory
        if hasattr(self, 'output_dir'):
            project_root = Path(self.output_dir).parent
            sqlite_path = project_root / f"{db_name}_db.db"
        else:
            sqlite_path = Path(f"{db_name}_db.db")

        print(f"   Creating SQLite database: {sqlite_path}")

        eng = create_engine(f"sqlite:///{sqlite_path}", echo=self.echo_sql, future=True)
        with eng.begin() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = ON;")
        return eng

    def _create_schema(self, engine: Engine, schema: Dict[str, Any]):
        md = MetaData()
        for e in schema["entities"]:
            cols=[]
            for a in e["attributes"]:
                cols.append(Column(a["name"], _sqlalchemy_type(a["type"])(), primary_key=(a["name"] in e["primary_key"])))
            Table(e["name"], md, *cols)
        md.create_all(engine)

    def _load_data(self, engine, schema, files):
        import pandas as pd
        from pathlib import Path
        from sqlalchemy import MetaData, Table

        # Build a quick type map: table -> {col -> logical_type}
        table_type_map = {
            e["name"]: {a["name"]: a["type"] for a in e["attributes"]}
            for e in schema["entities"]
        }

        # Map normalized filename -> path
        def _norm(name):
            import re
            from pathlib import Path
            s = Path(name).stem
            s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
            s = re.sub(r"__+", "_", s).strip("_").lower()
            return s

        file_map = {_norm(Path(f).stem): f for f in files}
        report = {}

        with engine.begin() as conn:
            for e in schema["entities"]:
                name = e["name"]
                csv_path = file_map.get(name)
                if not csv_path:
                    report[name] = {"rows": 0, "status": "no_csv"}
                    continue

                df = pd.read_csv(csv_path)
                # normalize columns to snake_case used in schema creation
                def _to_snake(s):
                    import re
                    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(s))
                    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
                    s = re.sub(r"__+", "_", s).strip("_").lower()
                    return s
                df.columns = [_to_snake(c) for c in df.columns]

                # Coerce types to match the DDL for this table
                tmap = table_type_map.get(name, {})
                for col, t in tmap.items():
                    if col not in df.columns:
                        # add missing nullable columns as NA
                        df[col] = pd.NA
                        continue

                    if t == "date":
                        s = pd.to_datetime(df[col], errors="coerce")
                        df[col] = s.dt.date  # Python datetime.date objects
                    elif t == "datetime":
                        s = pd.to_datetime(df[col], errors="coerce", utc=False)
                        # ensure Python datetime.datetime (naive)
                        df[col] = s.dt.tz_localize(None).apply(
                            lambda x: x.to_pydatetime() if pd.notna(x) else None
                        )
                    elif t == "boolean":
                        df[col] = (
                            df[col]
                            .replace(
                                {
                                    "true": True, "false": False,
                                    "True": True, "False": False,
                                    "1": True, "0": False,
                                    1: True, 0: False
                                }
                            )
                            .astype("boolean")
                            .astype(object)  # ensure Python bool/None
                        )
                    # numeric coercions are usually fine; optional stricter casts:
                    elif t in ("integer", "bigint"):
                        # df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                        import numpy as np
                        vals = pd.to_numeric(df[col], errors="coerce")

                        # If any fractional values appear, round them (so inserts won't fail)
                        frac = vals.dropna() - np.floor(vals.dropna())
                        if (np.abs(frac) > 1e-9).any():
                            self.logger.warning(f"[{name}.{col}] fractional numbers found in an integer column; rounding to nearest int.")
                            vals = vals.round()
                        # Use Python ints (object dtype) to avoid Int64 casting issues
                        df[col] = vals.apply(lambda x: int(x) if pd.notna(x) else None).astype(object)


                    elif t == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Reorder to table columns and convert NaN->None
                table_cols = [a["name"] for a in e["attributes"]]
                for c in table_cols:
                    if c not in df.columns:
                        df[c] = pd.NA
                df = df[table_cols]
                records = df.where(pd.notna(df), None).to_dict(orient="records")

                conn.execute(Table(name, MetaData(), autoload_with=engine).insert(), records)
                report[name] = {"rows": len(records), "status": "loaded"}

        return report

class RequirementsAgent:
    """Analyzes and structures requirements"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Extract and structure requirements"""
        print("📋 Analyzing Requirements...")

        # CRITICAL: Preserve any inferred entities from previous steps
        preserved_entities = state.get("inferred_entities")
        if preserved_entities:
            print(f"   RequirementsAgent: Preserving inferred entities: {preserved_entities}")

        # Extract requirements from messages
        messages = state.get("messages", [])
        requirements_text = ""
        entities = []

        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                requirements_text += content + "\n"

                # Extract entities if present
                if "ENTITIES:" in content:
                    for line in content.split('\n'):
                        if "ENTITIES:" in line:
                            entity_text = line.split("ENTITIES:")[1].strip()
                            entities = [e.strip() for e in entity_text.split(',')]
                            break

        # Structure requirements using LLM
        structured_requirements = self._structure_requirements(requirements_text)

        # CRITICAL: If we had inferred entities, use those instead
        if preserved_entities:
            structured_requirements["entities"] = preserved_entities
            print(f"   RequirementsAgent: Added preserved entities to requirements: {preserved_entities}")
        elif entities:
            structured_requirements["entities"] = entities

        # Save to files
        if self.artifact_manager:
            self.artifact_manager.save_requirements(structured_requirements)

        # CRITICAL: Update state WITHOUT losing other keys
        state["requirements"] = structured_requirements

        # Make sure inferred_entities is still in state
        if preserved_entities:
            state["inferred_entities"] = preserved_entities
            print(f"   RequirementsAgent: Confirmed inferred_entities still in state: {preserved_entities}")

        return state

    def _structure_requirements(self, requirements_text: str) -> Dict[str, Any]:
        """Structure requirements using LLM"""
        prompt_parts = [
            "Analyze the following requirements and structure them into categories:",
            "",
            "Requirements Text:",
            requirements_text,
            "",
            "Please structure these requirements into the following categories:",
            "1. Functional Requirements - what the system should do",
            "2. Non-Functional Requirements - performance, scalability, etc.",
            "3. Data Requirements - what data needs to be managed",
            "4. Integration Requirements - external system connections",
            "5. Compliance Requirements - regulatory and legal needs",
            "",
            "Return as JSON format with arrays for each category.",
            "Example:",
            '{',
            '  "functional_requirements": ["User management", "Data processing"],',
            '  "non_functional_requirements": ["High performance", "Scalability"],',
            '  "data_requirements": ["User data", "Transaction data"],',
            '  "integration_requirements": ["API integration", "Database connectivity"],',
            '  "compliance_requirements": ["Data privacy", "Audit trails"]',
            '}'
        ]

        prompt = "\n".join(prompt_parts)

        try:
            if self.llm:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Try to parse JSON from response
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

            # Fallback: analyze text and create basic structure
            return self._create_basic_requirements_structure(requirements_text)

        except Exception as e:
            print(f"   ⚠️ LLM analysis failed, using basic structure: {e}")
            return self._create_basic_requirements_structure(requirements_text)

    def _create_basic_requirements_structure(self, requirements_text: str) -> Dict[str, Any]:
        """Create basic requirements structure from text analysis"""
        text_lower = requirements_text.lower()

        # Check if this is the inventory management demo
        if "inventory management system" in text_lower and "multi-location retail" in text_lower:
            return {
                "functional_requirements": [
                    "Product catalog management",
                    "Multi-warehouse inventory tracking",
                    "Order fulfillment",
                    "Supplier management",
                    "Customer management"
                ],
                "non_functional_requirements": ["High performance", "Scalability", "Security"],
                "data_requirements": [
                    "Product data",
                    "Inventory data",
                    "Order data",
                    "Warehouse data",
                    "Supplier data",
                    "Customer data"
                ],
                "integration_requirements": ["API integration", "ERP integration"],
                "compliance_requirements": ["Audit trails", "Data privacy"],
                "entities": [
                    "Product", "Inventory", "Warehouse", "Supplier",
                    "Order", "OrderItem", "Customer", "Category", "Location"
                ]
            }

        # Simple keyword-based analysis
        functional = []
        if "inventory" in text_lower: functional.append("Inventory management")
        if "user" in text_lower: functional.append("User management")
        if "order" in text_lower: functional.append("Order processing")
        if "product" in text_lower: functional.append("Product management")
        if "report" in text_lower: functional.append("Reporting and analytics")

        non_functional = []
        if "performance" in text_lower: non_functional.append("High performance")
        if "scale" in text_lower: non_functional.append("Scalability")
        if "security" in text_lower: non_functional.append("Security")
        if "availability" in text_lower: non_functional.append("High availability")

        data_reqs = []
        if "customer" in text_lower: data_reqs.append("Customer data")
        if "product" in text_lower: data_reqs.append("Product data")
        if "inventory" in text_lower: data_reqs.append("Inventory data")
        if "transaction" in text_lower: data_reqs.append("Transaction data")

        integration = []
        if "api" in text_lower: integration.append("API integration")
        if "erp" in text_lower: integration.append("ERP system integration")
        if "e-commerce" in text_lower: integration.append("E-commerce platform integration")

        compliance = []
        if "audit" in text_lower: compliance.append("Audit trails")
        if "gdpr" in text_lower: compliance.append("GDPR compliance")
        if "privacy" in text_lower: compliance.append("Data privacy")

        return {
            "functional_requirements": functional or ["Basic system functionality"],
            "non_functional_requirements": non_functional or ["Standard performance requirements"],
            "data_requirements": data_reqs or ["Core business data"],
            "integration_requirements": integration or ["Standard integrations"],
            "compliance_requirements": compliance or ["Basic compliance requirements"]
        }

# ============================================================================
# CONCEPTUAL MODEL AGENT
# ============================================================================


class ConceptualModelAgent:
    """Creates conceptual data models"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create conceptual model from requirements"""
        print("🎨 Creating Conceptual Model...")

        # Debug: Check what's in state
        print(f"   ConceptualModelAgent: Checking state for entities...")

        requirements = state.get("requirements", {})

        # Check for inferred entities FIRST at the state level
        inferred_entities = state.get("inferred_entities")

        if inferred_entities:
            print(f"   ✅ Found inferred_entities in state: {', '.join(inferred_entities)}")
            # Make sure requirements has these entities
            if not requirements:
                requirements = {}
            requirements["entities"] = inferred_entities
        elif requirements.get("entities"):
            print(f"   ✅ Using entities from requirements: {', '.join(requirements['entities'])}")
        else:
            print(f"   ⚠️ WARNING - No entities found, will have to extract")

        if not requirements and not inferred_entities:
            state["errors"].append("No requirements or inferred entities available")
            return state

        # Generate conceptual model
        xmi_model = self._generate_conceptual_model(requirements)

        # Generate DBML version
        dbml_model = self._generate_dbml_from_conceptual(xmi_model)

        # Save to files
        if self.artifact_manager:
            self.artifact_manager.save_model(xmi_model, "conceptual")
            self.artifact_manager.save_dbml(dbml_model, "conceptual")

        state["conceptual_model"] = xmi_model
        state["conceptual_model_dbml"] = dbml_model
        state["current_step"] = "conceptual_complete"

        # Preserve entities for next agents
        if inferred_entities:
            state["inferred_entities"] = inferred_entities

        return state

    def _generate_conceptual_model(self, requirements: Dict[str, Any]) -> str:
        # Use explicitly passed entities if available
        if "entities" in requirements and requirements["entities"]:
            entities = requirements["entities"]
            print(f"   Using provided entities: {', '.join(entities)}")
        else:
            entities = self._extract_entities_from_requirements(requirements)
            print(f"   Extracted entities: {', '.join(entities)}")

        # Make sure we have valid entities
        if not entities or entities == ["Entity1", "Entity2", "Entity3"]:
            print(f"   WARNING: No valid entities found, using fallback")
            entities = ["MainEntity", "SecondaryEntity", "Configuration"]

        prompt_parts = [
            "Create a conceptual data model in XMI format based on these requirements:",
            "",
            "Requirements:",
            json.dumps(requirements, indent=2),
            "",
            f"MANDATORY: You MUST include EXACTLY these {len(entities)} entities:",
            "\n".join([f"{i+1}. {entity}" for i, entity in enumerate(entities)]),
            "",
            "Instructions:",
            f"1. Create a conceptual model with EXACTLY {len(entities)} entities listed above",
            "2. DO NOT skip any entities from the list",
            "3. DO NOT add entities not in the list",
            "4. Include the following standard attributes for ALL entities:",
            "   - id (identifier/primary key)",
            "   - name (string, required)",
            "   - description (text, optional)",
            "   - created_at (datetime for tracking creation)",
            "   - updated_at (datetime for tracking updates)",
            "5. Add additional relevant attributes for each entity",
            "6. Define meaningful relationships between entities",
            "",
            "Output format:",
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="ConceptualModel" type="Conceptual">',
            '    <entities>',
            '      <entity name="EntityName" description="Business description">',
            '        <attribute name="id" type="identifier"/>',
            '        <attribute name="name" type="string"/>',
            '        <attribute name="description" type="text"/>',
            '        <attribute name="created_at" type="datetime"/>',
            '        <attribute name="updated_at" type="datetime"/>',
            '        <!-- Additional attributes here -->',
            '      </entity>',
            '    </entities>',
            '    <relationships>',
            '      <relationship name="RelationshipName" from="Entity1" to="Entity2" cardinality="one-to-many"/>',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>',
            "",
            "Generate the complete XMI model:"
        ]

        prompt = "\n".join(prompt_parts)

        try:
            if self.llm:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)
                xmi_content = self._extract_xmi_from_response(response_content)
                if self._validate_xmi(xmi_content):
                    return xmi_content

            return self._create_fallback_conceptual_model(entities)

        except Exception as e:
            print(f"   ⚠️ LLM model generation failed, using fallback: {e}")
            return self._create_fallback_conceptual_model(entities)

    def _extract_entities_from_requirements(self, requirements: Dict[str, Any]) -> List[str]:
        """Extract potential entities from requirements - only as last resort"""

        # If we already have entities, don't extract more
        if requirements.get("inferred_entities"):
            return requirements["inferred_entities"]
        if requirements.get("entities"):
            return requirements["entities"]

        entities = set()

        # Only look in data_requirements, not all requirements
        data_reqs = requirements.get('data_requirements', [])
        for req in data_reqs:
            if isinstance(req, str):
                # Only extract nouns that look like entity names
                words = re.findall(r'\b([A-Z][a-z]+)\b', req)
                # Filter out common non-entity words more aggressively
                filtered = [w for w in words if w not in {
                    'The', 'This', 'That', 'System', 'Data', 'Management',
                    'Support', 'Integration', 'Compliance', 'Scalability',
                    'Performance', 'Security', 'Requirements', 'Functional',
                    'Non', 'High', 'Standard', 'Basic', 'Core'
                }]
                entities.update(filtered)

        # Add some domain-specific entities based on keywords if found
        all_text = " ".join(str(req) for req_list in requirements.values()
                          if isinstance(req_list, list) for req in req_list)

        if "inventory" in all_text.lower() and not entities:
            entities.update(['Product', 'Inventory', 'Warehouse', 'Supplier'])
        if "customer" in all_text.lower() and not entities:
            entities.update(['Customer', 'Order'])
        if "user" in all_text.lower() and not entities:
            entities.update(['User', 'Role'])

        return list(entities) if entities else ["Entity1", "Entity2", "Entity3"]

    def _extract_xmi_from_response(self, response: str) -> str:
        """Extract XMI from LLM response"""
        if "<?xml" in response:
            start = response.find("<?xml")
            end_markers = ["</xmi:XMI>", "</model>"]
            end = -1
            for marker in end_markers:
                marker_pos = response.find(marker, start)
                if marker_pos != -1:
                    end = marker_pos + len(marker)
                    break

            if end != -1:
                return response[start:end]

        return response

    def _validate_xmi(self, xmi_content: str) -> bool:
        """Validate XMI content"""
        try:
            root = ET.fromstring(xmi_content)
            entities = root.findall(".//entity") or root.findall(".//Entity")
            return len(entities) >= 3
        except ET.ParseError:
            return False

    def _create_fallback_conceptual_model(self, entities: List[str]) -> str:
        """Create fallback conceptual model"""

        # Use only the entities we actually have, don't add extras
        if not entities:
            entities = ["MainEntity"]

        xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="ConceptualModel" type="Conceptual">',
            '    <entities>'
        ]

        # Create an entity for each actual entity only
        for entity in entities:
            xmi_parts.extend([
                f'      <entity name="{entity}" description="{entity} business entity">',
                '        <attribute name="id" type="identifier"/>',
                '        <attribute name="name" type="string"/>',
                '        <attribute name="description" type="text"/>',
                '        <attribute name="created_at" type="datetime"/>',
                '        <attribute name="updated_at" type="datetime"/>',
                '        <attribute name="status" type="enumeration"/>',
                '      </entity>'
            ])

        xmi_parts.append('    </entities>')
        xmi_parts.append('    <relationships>')

        # Only add relationships if we have multiple entities
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                xmi_parts.append(
                    f'      <relationship name="{entities[i]}{entities[i+1]}Association" '
                    f'from="{entities[i]}" to="{entities[i+1]}" cardinality="one-to-many"/>'
                )

        xmi_parts.extend([
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(xmi_parts)

    def _generate_dbml_from_conceptual(self, conceptual_model: str) -> str:
        """Generate DBML representation from conceptual model XMI"""

        dbml_parts = [
            "// Conceptual Data Model in DBML format",
            "// Generated by Data Architect Agent",
            ""
        ]

        try:
            # Parse the XMI
            root = ET.fromstring(conceptual_model)

            # Extract entities
            entities = root.findall(".//entity") or root.findall(".//Entity")

            # Generate DBML for each entity
            for entity in entities:
                entity_name = entity.get("name")
                description = entity.get("description", "")

                # Start entity definition
                dbml_parts.append(f"Table {entity_name} {{")

                if description:
                    dbml_parts.append(f"  // {description}")

                # Standard attributes that should be present in every entity
                standard_attributes = {
                    "id": "integer [pk]",
                    "name": "string [not null]",
                    "description": "text",
                    "created_at": "datetime",
                    "updated_at": "datetime"
                }

                # Track which standard attributes we've seen
                found_standard_attrs = {attr: False for attr in standard_attributes}

                # Add attributes
                attributes = entity.findall("./attribute") or entity.findall("./Attribute")
                for attr in attributes:
                    attr_name = attr.get("name")
                    attr_type = attr.get("type", "string")

                    # Check if this is a standard attribute
                    is_standard = attr_name in standard_attributes
                    if is_standard:
                        found_standard_attrs[attr_name] = True

                    # Add ALL attributes (both standard and non-standard)
                    dbml_type = self._map_to_dbml_type(attr_type)
                    dbml_parts.append(f"  {attr_name} {dbml_type}")

                # Add any standard attributes that weren't found
                for attr_name, found in found_standard_attrs.items():
                    if not found:
                        dbml_parts.append(f"  {attr_name} {standard_attributes[attr_name]}")

                # Close entity definition
                dbml_parts.append("}")
                dbml_parts.append("")

            # Extract relationships
            relationships = root.findall(".//relationship") or root.findall(".//Relationship")

            # Add relationships
            if relationships:
                dbml_parts.append("// Relationships")

                for rel in relationships:
                    from_entity = rel.get("from")
                    to_entity = rel.get("to")
                    cardinality = rel.get("cardinality", "").lower()

                    # Map cardinality to DBML syntax
                    if "many-to-many" in cardinality:
                        dbml_rel = "<>"
                    elif "one-to-many" in cardinality:
                        dbml_rel = ">"
                    elif "many-to-one" in cardinality:
                        dbml_rel = "<"
                    else:
                        dbml_rel = "-"  # Default to one-to-one

                    # Add the relationship - always use id as the reference column
                    dbml_parts.append(f"Ref: {from_entity}.id {dbml_rel} {to_entity}.id")

                dbml_parts.append("")

        except Exception as e:
            # Fallback: create a basic DBML structure
            dbml_parts = [
                "// Conceptual Data Model in DBML format",
                "// Generated by Data Architect Agent",
                "// Note: This is a simplified representation due to parsing error",
                f"// Error: {str(e)}",
                "",
                "Table Entity1 {",
                "  id integer [pk]",
                "  name string [not null]",
                "  description text",
                "  created_at datetime",
                "  updated_at datetime",
                "  status string",
                "}",
                "",
                "Table Entity2 {",
                "  id integer [pk]",
                "  name string [not null]",
                "  description text",
                "  created_at datetime",
                "  updated_at datetime",
                "  type string",
                "}",
                "",
                "Ref: Entity1.id - Entity2.id"
            ]

        return "\n".join(dbml_parts)

    def _map_to_dbml_type(self, xmi_type: str) -> str:
        """Map XMI data types to DBML data types"""
        type_map = {
            "identifier": "integer [pk]",
            "id": "integer [pk]",
            "string": "string",
            "text": "text",
            "integer": "integer",
            "int": "integer",
            "number": "float",
            "float": "float",
            "decimal": "decimal",
            "boolean": "boolean",
            "date": "date",
            "datetime": "datetime",
            "timestamp": "timestamp",
            "enumeration": "string"
        }

        # Default to string if type is not recognized
        return type_map.get(xmi_type.lower(), "string")

# ============================================================================
# LOGICAL MODEL AGENT
# ============================================================================

class LogicalModelAgent:
    """Creates logical data models"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create logical model from conceptual model"""
        print("🔗 Creating Logical Model...")

        # Debug: Check for entities
        print(f"   LogicalModelAgent: Checking for entities in state...")
        print(f"   - inferred_entities: {state.get('inferred_entities')}")

        conceptual_model = state.get("conceptual_model")
        if not conceptual_model:
            state["errors"].append("No conceptual model available")
            return state

        # Pass requirements with entities if available
        requirements = state.get("requirements", {})
        if state.get("inferred_entities"):
            requirements["entities"] = state["inferred_entities"]

        # Generate logical model
        logical_model = self._generate_logical_model(conceptual_model, requirements)

        # Generate DBML version
        dbml_model = self._generate_dbml_from_logical(logical_model)

        # Save to files
        if self.artifact_manager:
            self.artifact_manager.save_model(logical_model, "logical")
            self.artifact_manager.save_dbml(dbml_model, "logical")

        state["logical_model"] = logical_model
        state["logical_model_dbml"] = dbml_model
        state["current_step"] = "logical_complete"
        return state

    def _generate_logical_model(self, conceptual_model: str, requirements: Dict[str, Any]) -> str:
        """Generate logical model"""

        def _pluralize(noun: str) -> str:
            noun = noun.strip()
            lower = noun.lower()
            # Basic rules that cover 95% of what we need here
            if lower.endswith(("s", "x", "z", "ch", "sh")):
                return lower + "es"
            if lower.endswith("y") and (len(lower) > 1 and lower[-2] not in "aeiou"):
                return lower[:-1] + "ies"
            return lower + "s"


        # Extract entities from conceptual model
        entities = self._extract_entities_from_conceptual(conceptual_model)

        # Create logical model XMI
        logical_xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="LogicalModel" type="Logical">',
            '    <tables>'
        ]

        # Create tables for each entity
        for entity in entities:
            # table_name = entity.lower() + "s"  # Pluralize for table names
            table_name = _pluralize(entity)
            logical_xmi_parts.extend([
                f'      <table name="{table_name}">',
                f'        <column name="id" type="integer" primary_key="true"/>',
                f'        <column name="name" type="varchar(255)" nullable="false"/>',
                f'        <column name="description" type="text" nullable="true"/>',
                f'        <column name="created_at" type="timestamp" default="CURRENT_TIMESTAMP"/>',
                f'        <column name="updated_at" type="timestamp"/>'
            ])

            # Add entity-specific columns based on entity type/name
            if entity.lower() == "user":
                logical_xmi_parts.extend([
                    f'        <column name="email" type="varchar(255)" nullable="false"/>',
                    f'        <column name="password_hash" type="varchar(255)" nullable="false"/>',
                    f'        <column name="status" type="varchar(50)" default="\'active\'"/>'
                ])
            elif entity.lower() == "product":
                logical_xmi_parts.extend([
                    f'        <column name="sku" type="varchar(100)" nullable="false"/>',
                    f'        <column name="price" type="decimal(10,2)"/>',
                    f'        <column name="category_id" type="integer"/>'
                ])
            elif entity.lower() == "order":
                logical_xmi_parts.extend([
                    f'        <column name="order_date" type="date"/>',
                    f'        <column name="customer_id" type="integer"/>',
                    f'        <column name="status" type="varchar(50)"/>'
                ])

            # Close the table tag
            logical_xmi_parts.append(f'      </table>')

        logical_xmi_parts.extend([
            '    </tables>',
            '    <relationships>',
            '      <!-- Foreign key relationships will be defined here -->',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(logical_xmi_parts)

    def _extract_entities_from_conceptual(self, conceptual_model: str) -> List[str]:
        """Extract entity names from conceptual model"""
        entities = []

        # Parse XMI and extract entity names
        try:
            root = ET.fromstring(conceptual_model)
            for entity in root.findall(".//entity"):
                name = entity.get("name")
                if name:
                    entities.append(name)
        except:
            # Fallback: extract from text
            entity_matches = re.findall(r'name="([^"]+)"', conceptual_model)
            entities = [match for match in entity_matches if match not in ['ConceptualModel', 'LogicalModel']]

        return entities or ["MainEntity", "SecondaryEntity", "Configuration"]

    def _generate_dbml_from_logical(self, logical_model: str) -> str:
        """Generate DBML representation from logical model XMI"""

        dbml_parts = [
            "// Logical Data Model in DBML format",
            "// Generated by Data Architect Agent",
            ""
        ]

        try:
            # Parse the XMI
            root = ET.fromstring(logical_model)

            # Extract tables
            tables = root.findall(".//table") or root.findall(".//Table")

            # Generate DBML for each table
            for table in tables:
                table_name = table.get("name")

                # Start table definition
                dbml_parts.append(f"Table {table_name} {{")

                # Standard columns that should be present in every table
                standard_columns = {
                    "id": "integer [pk]",
                    "name": "varchar [not null]",
                    "description": "text",
                    "created_at": "timestamp",
                    "updated_at": "timestamp"
                }

                # Track which standard columns we've seen
                found_standard_cols = {col: False for col in standard_columns}

                # Add columns
                columns = table.findall("./column") or table.findall("./Column")
                for col in columns:
                    col_name = col.get("name")
                    col_type = col.get("type", "varchar")
                    is_pk = col.get("primary_key") == "true"
                    nullable = col.get("nullable") != "false"
                    default = col.get("default", "")

                    # Check if this is a standard column
                    is_standard = col_name in standard_columns
                    if is_standard:
                        found_standard_cols[col_name] = True

                    # Format the column definition - handle ALL columns
                    if is_pk:
                        dbml_parts.append(f"  {col_name} {col_type} [pk]")
                    else:
                        # Add ALL non-PK columns including standard ones
                        null_str = "" if nullable else " [not null]"
                        default_str = f" [default: '{default}']" if default else ""
                        dbml_parts.append(f"  {col_name} {col_type}{null_str}{default_str}")

                # Add any standard columns that weren't found at all
                for col_name, found in found_standard_cols.items():
                    if not found:
                        dbml_parts.append(f"  {col_name} {standard_columns[col_name]}")

                # Close table definition
                dbml_parts.append("}")
                dbml_parts.append("")

        # Rest of the method remains the same...

            # Extract relationships
            relationships = root.findall(".//relationship") or root.findall(".//Relationship")

            # Add relationships
            if relationships:
                dbml_parts.append("// Relationships")

                for rel in relationships:
                    from_table = rel.get("from_table") or rel.get("from")
                    to_table = rel.get("to_table") or rel.get("to")

                    if from_table and to_table:
                        # Always reference id columns for relationships
                        dbml_parts.append(f"Ref: {from_table}.id > {to_table}.id")

                dbml_parts.append("")

        except Exception as e:
            # Fallback: create a basic DBML structure
            dbml_parts = [
                "// Logical Data Model in DBML format",
                "// Generated by Data Architect Agent",
                "// Note: This is a simplified representation due to parsing error",
                f"// Error: {str(e)}",
                "",
                "Table main_table {",
                "  id integer [pk]",
                "  name varchar [not null]",
                "  description text",
                "  created_at timestamp",
                "  updated_at timestamp",
                "  status varchar",
                "}",
                "",
                "Table related_table {",
                "  id integer [pk]",
                "  name varchar [not null]",
                "  description text",
                "  created_at timestamp",
                "  updated_at timestamp",
                "  main_id integer [not null]",
                "}",
                "",
                "Ref: related_table.main_id > main_table.id"
            ]

        return "\n".join(dbml_parts)

# ============================================================================
# PHYSICAL MODEL AGENT
# ============================================================================

class PhysicalModelAgent:
    """Creates physical models and SQL DDL"""

    def __init__(self, llm, target_database="postgresql", artifact_manager=None):
        self.llm = llm
        self.target_database = target_database
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create physical model and SQL DDL"""
        print("🗄️ Creating Physical Model and SQL DDL...")

        logical_model = state.get("logical_model")
        if not logical_model:
            state["errors"].append("No logical model available")
            return state

        # Generate DDL
        physical_artifacts = self._generate_physical_artifacts(logical_model, state.get("requirements"))

        # Save DDL
        if self.artifact_manager:
            self.artifact_manager.save_implementation_artifacts(physical_artifacts)
            self.artifact_manager.save_model(physical_artifacts["xmi_model"], "physical")

        state["physical_model"] = physical_artifacts["xmi_model"]
        state["sql_ddl"] = physical_artifacts["sql_ddl"]  # Store for easy access

        if not state.get("implementation_artifacts"):
            state["implementation_artifacts"] = {}

        state["implementation_artifacts"]["sql_ddl"] = physical_artifacts["sql_ddl"]
        state["implementation_artifacts"]["indexes"] = physical_artifacts["indexes"]
        state["implementation_artifacts"]["constraints"] = physical_artifacts["constraints"]

        state["current_step"] = "physical_complete"
        return state

    def _generate_physical_artifacts(self, logical_model: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate physical model artifacts"""

        # Extract table information from logical model
        tables = self._extract_tables_from_logical(logical_model)

        # Generate SQL DDL
        sql_ddl = self._generate_sql_ddl(tables, requirements)

        # Extract indexes and constraints
        indexes = self._extract_indexes_from_ddl(sql_ddl)
        constraints = self._extract_constraints_from_ddl(sql_ddl)

        # Create physical XMI
        xmi_model = self._create_physical_xmi(sql_ddl)

        return {
            "xmi_model": xmi_model,
            "sql_ddl": sql_ddl,
            "indexes": indexes,
            "constraints": constraints
        }

    def _extract_tables_from_logical(self, logical_model: str) -> List[str]:
        """Extract table names from logical model"""
        tables = []

        try:
            root = ET.fromstring(logical_model)
            for table in root.findall(".//table"):
                name = table.get("name")
                if name:
                    tables.append(name)
        except:
            # Fallback: extract from text
            table_matches = re.findall(r'<table name="([^"]+)"', logical_model)
            tables = table_matches

        return tables or ["main_entities", "secondary_entities", "configurations"]

    def _generate_sql_ddl(self, tables: List[str], requirements: Dict[str, Any]) -> str:
        """Generate SQL DDL for tables"""

        ddl_parts = [
            f"-- {self.target_database.title()} Schema",
            f"-- Generated by Data Architect Agent",
            f"-- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-- Enable extensions",
            "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";",
            ""
        ]

        # Generate CREATE TABLE statements
        for table in tables:
            ddl_parts.extend(self._generate_table_ddl(table))
            ddl_parts.append("")

        # Generate indexes
        ddl_parts.append("-- Indexes for performance")
        for table in tables:
            ddl_parts.extend(self._generate_table_indexes(table))

        # Generate constraints
        ddl_parts.append("")
        ddl_parts.append("-- Additional constraints")
        for table in tables:
            ddl_parts.extend(self._generate_table_constraints(table))

        return "\n".join(ddl_parts)

    def _generate_table_ddl(self, table_name: str) -> List[str]:
        """Generate DDL for a single table"""

        # Standard columns for ALL tables
        standard_columns = [
            "    id SERIAL PRIMARY KEY,",
            "    name VARCHAR(255) NOT NULL,",
            "    description TEXT,",
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
            "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        ]

        # Start with standard columns
        columns = standard_columns.copy()

        # Determine additional columns based on table name patterns
        if "user" in table_name.lower():
            columns.extend([
                "    username VARCHAR(100) UNIQUE NOT NULL,",
                "    email VARCHAR(255) UNIQUE NOT NULL,",
                "    password_hash VARCHAR(255) NOT NULL,",
                "    first_name VARCHAR(100),",
                "    last_name VARCHAR(100),",
                "    status VARCHAR(20) DEFAULT 'active'"
            ])
        elif "product" in table_name.lower():
            columns.extend([
                "    sku VARCHAR(100) UNIQUE NOT NULL,",
                "    category_id INTEGER,",
                "    price DECIMAL(10,2),",
                "    weight DECIMAL(8,3),",
                "    status VARCHAR(20) DEFAULT 'active'"
            ])
        elif "inventory" in table_name.lower():
            columns.extend([
                "    product_id INTEGER NOT NULL,",
                "    location_id INTEGER,",
                "    quantity_on_hand INTEGER NOT NULL DEFAULT 0,",
                "    quantity_allocated INTEGER DEFAULT 0"
            ])
        elif "order" in table_name.lower():
            columns.extend([
                "    customer_id INTEGER NOT NULL,",
                "    order_date DATE NOT NULL,",
                "    status VARCHAR(50) DEFAULT 'pending',",
                "    total_amount DECIMAL(12,2),",
                "    payment_method VARCHAR(50)"
            ])
        elif "customer" in table_name.lower():
            columns.extend([
                "    email VARCHAR(255) UNIQUE,",
                "    phone VARCHAR(50),",
                "    address TEXT,",
                "    customer_since DATE,",
                "    status VARCHAR(20) DEFAULT 'active'"
            ])
        else:
            # For other tables, we already have the standard columns
            columns.extend([
                "    status VARCHAR(50) DEFAULT 'active',",
                "    metadata JSONB"
            ])
        # Ensure no trailing comma on the final column
        if columns:
            columns[-1] = columns[-1].rstrip().rstrip(',')
        ddl = [f"CREATE TABLE {table_name} ("] + columns + [");"]
        return ddl

    def _generate_table_indexes(self, table_name: str) -> List[str]:
        """Generate indexes for a table"""
        indexes = [
            f"CREATE INDEX idx_{table_name}_created_at ON {table_name}(created_at);",
            f"CREATE INDEX idx_{table_name}_status ON {table_name}(status);"
        ]

        # Add specific indexes based on table type
        if "user" in table_name.lower():
            indexes.append(f"CREATE INDEX idx_{table_name}_email ON {table_name}(email);")
        elif "product" in table_name.lower():
            indexes.append(f"CREATE INDEX idx_{table_name}_sku ON {table_name}(sku);")
            indexes.append(f"CREATE INDEX idx_{table_name}_name ON {table_name}(name);")
        elif "inventory" in table_name.lower():
            indexes.append(f"CREATE INDEX idx_{table_name}_product_id ON {table_name}(product_id);")
            indexes.append(f"CREATE INDEX idx_{table_name}_location_id ON {table_name}(location_id);")
        else:
            indexes.append(f"CREATE INDEX idx_{table_name}_name ON {table_name}(name);")

        return indexes

    def _generate_table_constraints(self, table_name: str) -> List[str]:
        """Generate constraints for a table"""
        constraints = []

        if "inventory" in table_name.lower():
            constraints.extend([
                f"ALTER TABLE {table_name} ADD CONSTRAINT chk_{table_name}_quantity_positive",
                f"    CHECK (quantity_on_hand >= 0 AND quantity_allocated >= 0);",
                f"ALTER TABLE {table_name} ADD CONSTRAINT chk_{table_name}_allocation_valid",
                f"    CHECK (quantity_allocated <= quantity_on_hand);"
            ])

        return constraints

    def _extract_indexes_from_ddl(self, sql_ddl: str) -> List[str]:
        """Extract CREATE INDEX statements"""
        indexes = []
        for line in sql_ddl.split('\n'):
            if 'CREATE INDEX' in line.upper():
                indexes.append(line.strip())
        return indexes

    def _extract_constraints_from_ddl(self, sql_ddl: str) -> List[str]:
        """Extract constraint definitions"""
        constraints = []
        for line in sql_ddl.split('\n'):
            if any(keyword in line.upper() for keyword in ['ALTER TABLE', 'ADD CONSTRAINT', 'CHECK']):
                constraints.append(line.strip())
        return constraints

    def _create_physical_xmi(self, sql_ddl: str) -> str:
        """Create XMI representation of physical model"""

        table_count = sql_ddl.upper().count('CREATE TABLE')

        xmi_template = '''<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">
  <model name="PhysicalModel" type="Physical" target_database="{target_db}">
    <description>Physical data model with {table_count} tables</description>
    <sql_ddl><![CDATA[
{sql_ddl}
    ]]></sql_ddl>
  </model>
</xmi:XMI>'''

        return xmi_template.format(
            target_db=self.target_database,
            table_count=table_count,
            sql_ddl=sql_ddl
        )

# ============================================================================
# GLOSSARY AGENT
# ============================================================================

class GlossaryAgent:
    """Creates a glossary of domain-specific terms with definitions"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create glossary from requirements and models"""
        print("📘 Creating Domain Glossary...")

        requirements = state.get("requirements")
        conceptual_model = state.get("conceptual_model")

        if not requirements:
            state["errors"].append("No requirements available for glossary creation")
            return state

        # Generate glossary
        glossary = self._generate_glossary(requirements, conceptual_model)

        # Save to files
        if self.artifact_manager:
            self.artifact_manager.save_glossary(glossary)

        state["glossary"] = glossary
        state["current_step"] = "glossary_complete"
        return state

    def _generate_glossary(self, requirements, conceptual_model=None):
        """Generate glossary of terms with definitions"""

        # Extract potential terms from requirements and conceptual model
        terms = self._extract_terms(requirements, conceptual_model)

        # Use LLM to generate definitions for each term
        glossary_entries = self._define_terms(terms, requirements)

        return {
            "title": "Domain Glossary",
            "description": "Glossary of domain-specific terms and definitions",
            "entries": glossary_entries
        }

    def _extract_terms(self, requirements, conceptual_model=None):
        """Extract potential glossary terms from requirements and conceptual model"""

        # Extract from requirements
        all_text = ""
        for req_list in requirements.values():
            if isinstance(req_list, list):
                all_text += " ".join(str(req) for req in req_list)

        # Extract potential terms using NLP techniques or LLM
        if self.llm:
            prompt_parts = [
                "Extract domain-specific terms from the following requirements that should be included in a glossary:",
                "",
                "Requirements:",
                all_text,
                "",
                "Extract only domain-specific technical terms, business concepts, or specialized vocabulary that would benefit from a clear definition.",
                "Return as a JSON array of strings, containing just the terms (without definitions yet).",
                "Example: [\"Inventory\", \"SKU\", \"Reorder Point\", \"ABC Classification\"]"
            ]

            prompt = "\n".join(prompt_parts)

            try:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Try to parse JSON array from response
                import json
                import re

                json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

            except Exception as e:
                print(f"   ⚠️ LLM term extraction failed: {e}")
                # Continue with basic extraction

        # Basic extraction fallback
        return self._basic_term_extraction(all_text, conceptual_model)

    def _basic_term_extraction(self, text, conceptual_model):
        """Basic term extraction using regex and domain knowledge"""
        import re

        # Extract capitalized phrases (potential terms)
        capitalized_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

        # Extract technical terms based on domain
        domain_terms = set()

        # Add terms based on domain detection
        if "inventory" in text.lower():
            domain_terms.update([
                "SKU", "Inventory", "Stock", "Warehouse", "Reorder Point",
                "Safety Stock", "Lead Time", "Stockout", "Backorder",
                "ABC Classification", "Cycle Count", "Physical Inventory",
                "FIFO", "LIFO", "EOQ", "Min/Max Levels"
            ])

        if "customer" in text.lower():
            domain_terms.update([
                "Customer", "Order", "Invoice", "Return", "Credit Memo",
                "Customer Lifetime Value", "Churn Rate", "Retention Rate"
            ])

        if "product" in text.lower():
            domain_terms.update([
                "Product", "Category", "Attribute", "Variant", "Bundle",
                "Bill of Materials", "Kit", "Assembly"
            ])

        # Add entities from conceptual model if available
        model_terms = set()
        if conceptual_model:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(conceptual_model)

                # Extract entity names
                for entity in root.findall(".//entity"):
                    name = entity.get("name")
                    if name:
                        model_terms.add(name)

                # Extract attribute names
                for attribute in root.findall(".//attribute"):
                    name = attribute.get("name")
                    if name:
                        model_terms.add(name.replace("_", " ").title())

            except:
                pass  # Silently continue if model parsing fails

        # Combine all terms and filter out common words
        all_terms = capitalized_terms.union(domain_terms).union(model_terms)
        filtered_terms = [
            term for term in all_terms
            if term.lower() not in [
                "the", "a", "an", "and", "or", "of", "in", "on", "with",
                "for", "to", "from", "by", "management", "system", "requirements"
            ]
        ]

        return sorted(list(set(filtered_terms)))

    def _define_terms(self, terms, requirements):
        """Generate definitions for each term using LLM"""

        # Get domain context from requirements
        domain_context = self._extract_domain_context(requirements)

        if self.llm and terms:
            prompt_parts = [
                f"Create a glossary for a {domain_context} system with definitions for the following terms:",
                "",
                "Terms:",
                ", ".join(terms),
                "",
                "Requirements context:",
                "\n".join(str(req) for req_list in requirements.values() for req in req_list if isinstance(req_list, list)),
                "",
                "For each term, provide a clear, concise definition in the context of this domain.",
                "Return as a JSON array of objects with 'term' and 'definition' properties.",
                "Example: [",
                "  {",
                "    \"term\": \"SKU\",",
                "    \"definition\": \"Stock Keeping Unit. A unique identifier assigned to a distinct product for inventory tracking.\"",
                "  },",
                "  {",
                "    \"term\": \"Reorder Point\",",
                "    \"definition\": \"The inventory level at which a new order should be placed to replenish stock.\"",
                "  }",
                "]"
            ]

            prompt = "\n".join(prompt_parts)

            try:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Try to parse JSON array from response
                import json
                import re

                json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

            except Exception as e:
                print(f"   ⚠️ LLM definition generation failed: {e}")
                # Fall back to basic definitions

        # Basic fallback definitions
        return self._create_basic_definitions(terms)

    def _extract_domain_context(self, requirements):
        """Extract domain context from requirements"""

        all_text = ""
        for req_list in requirements.values():
            if isinstance(req_list, list):
                all_text += " ".join(str(req) for req in req_list)

        text_lower = all_text.lower()

        if "inventory" in text_lower or "warehouse" in text_lower:
            return "Inventory Management"
        elif "ecommerce" in text_lower or "e-commerce" in text_lower or "shop" in text_lower:
            return "E-commerce"
        elif "customer" in text_lower and "relationship" in text_lower:
            return "CRM"
        elif "patient" in text_lower or "medical" in text_lower:
            return "Healthcare"
        elif "student" in text_lower or "education" in text_lower:
            return "Education"
        elif "financial" in text_lower or "bank" in text_lower:
            return "Financial Services"
        else:
            return "Business Management"

    def _create_basic_definitions(self, terms):
        """Create basic definitions for common terms"""

        basic_definitions = {
            "Inventory": "The goods and materials that a business holds for sale or production purposes.",
            "SKU": "Stock Keeping Unit. A unique identifier for a distinct product for inventory tracking.",
            "Product": "An item or service offered by a company, typically for sale.",
            "Order": "A request made by a customer to purchase goods or services.",
            "Customer": "An individual or organization that purchases goods or services.",
            "User": "An individual who interacts with the system, typically an employee or administrator.",
            "Warehouse": "A physical location where inventory is stored, managed, and fulfilled.",
            "Supplier": "A business entity that provides goods or services to another business.",
            "Transaction": "A recorded exchange of goods, services, or funds between parties.",
            "Category": "A grouping of similar products or items for organizational purposes.",
            "Attribute": "A property or characteristic of an entity or object in the system.",
            "API": "Application Programming Interface. A set of rules allowing different software to communicate.",
            "ERP": "Enterprise Resource Planning. An integrated management system for core business processes.",
            "CRM": "Customer Relationship Management. A system for managing customer interactions and data.",
            "Database": "An organized collection of data stored and accessed electronically.",
            "Schema": "The structure that defines how data is organized within a database.",
            "Entity": "A distinct object or concept about which data can be stored in a database.",
            "Relationship": "A connection or association between two or more entities in a data model."
        }

        glossary_entries = []
        for term in terms:
            if term in basic_definitions:
                glossary_entries.append({
                    "term": term,
                    "definition": basic_definitions[term]
                })
            else:
                glossary_entries.append({
                    "term": term,
                    "definition": f"A {term.lower()} in the context of this business domain."
                })

        return glossary_entries

# ============================================================================
# DATA PRODUCT AGENT
# ============================================================================


class DataProductAgent:
    """Creates data product specifications and APIs"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create data product artifacts"""
        print("📡 Creating Data Product APIs and Specifications...")

        physical_model = state.get("physical_model")
        if not physical_model:
            state["errors"].append("No physical model available")
            return state

        # Generate data product configuration
        data_product_config = self._generate_data_product_config(
            physical_model,
            state.get("requirements"),
            state.get("implementation_artifacts", {}),
            state.get("project_id")  # Pass the project_id
        )

        # Save to files
        if self.artifact_manager:
            self.artifact_manager.save_data_product_config(data_product_config)

        state["data_product_config"] = data_product_config
        state["current_step"] = "data_product_complete"
        return state

    # def _generate_data_product_config(self, physical_model: str, requirements: Dict[str, Any],
    #                                 implementation_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    def _generate_data_product_config(self, physical_model: str, requirements: Dict[str, Any],
                                implementation_artifacts: Dict[str, Any], project_id: str = None) -> Dict[str, Any]:
        """Generate data product configuration"""

        # Extract table names from SQL DDL
        sql_ddl = implementation_artifacts.get("sql_ddl", "")
        tables = self._extract_table_names_from_ddl(sql_ddl)

        # # Create API specification
        # api_spec = self._create_api_specification(tables, requirements)
        # Create API specification - pass project_id
        api_spec = self._create_api_specification(tables, requirements, project_id)

        # Create other data product components
        return {
            "api_specification": api_spec,
            "collibra_catalog": self._create_data_catalog(tables, requirements),
            "unity_catalog": self._create_unity_catalog(tables),
            "data_lineage": self._create_data_lineage(requirements),
            "api_documentation": self._create_api_documentation(api_spec),
            "governance_policies": self._create_governance_policies(requirements)
        }

    def _extract_table_names_from_ddl(self, sql_ddl: str) -> List[str]:
        """Extract table names from SQL DDL"""
        tables = []
        pattern = r'CREATE\s+TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, sql_ddl, re.IGNORECASE)
        return matches or ["items", "categories", "users"]

    # def _create_api_specification(self, tables: List[str], requirements: Dict[str, Any]) -> Dict[str, Any]:
    #     """Create OpenAPI 3.0 specification"""

    #     # Infer domain from requirements
    #     domain = self._infer_domain_from_requirements(requirements)
    def _create_api_specification(self, tables: List[str], requirements: Dict[str, Any], project_id: str = None) -> Dict[str, Any]:
        """Create OpenAPI 3.0 specification"""

        # Extract domain from project_id if available, otherwise infer from requirements
        domain = "Business Management"  # ADD THIS DEFAULT
        if project_id:
            # Extract domain from project_id by removing suffixes
            domain = project_id.replace('_architecture', '').replace('_reverse_engineered', '')
            domain = domain.replace('_', ' ').title()
        elif requirements:  # CHANGE: Add elif check
            # Fallback to inferring from requirements
            domain = self._infer_domain_from_requirements(requirements)
        # If both are None, domain remains "Business Management"

        paths = {}
        schemas = {}

        # Create CRUD endpoints for each table
        for table in tables[:6]:  # Limit to avoid overwhelming
            # Create path names
            resource_path = "/" + table
            item_path = "/" + table + "/{id}"

            # Singular form for schema
            schema_name = table.rstrip('s').title() if table.endswith('s') else table.title()

            # List and create endpoints
            paths[resource_path] = {
                "get": {
                    "summary": "List " + table,
                    "description": "Retrieve a paginated list of " + table,
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
                        {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
                        {"name": "search", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "List of " + table + " with pagination"}
                    }
                },
                "post": {
                    "summary": "Create " + schema_name.lower(),
                    "description": "Create a new " + schema_name.lower(),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/" + schema_name}
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": schema_name + " created successfully"},
                        "400": {"description": "Invalid input data"}
                    }
                }
            }

            # Individual item endpoints
            paths[item_path] = {
                "get": {
                    "summary": "Get " + schema_name.lower(),
                    "description": "Retrieve a specific " + schema_name.lower() + " by ID",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {"description": schema_name + " details"},
                        "404": {"description": schema_name + " not found"}
                    }
                },
                "put": {
                    "summary": "Update " + schema_name.lower(),
                    "description": "Update an existing " + schema_name.lower(),
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/" + schema_name}
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": schema_name + " updated successfully"},
                        "404": {"description": schema_name + " not found"}
                    }
                },
                "delete": {
                    "summary": "Delete " + schema_name.lower(),
                    "description": "Delete a " + schema_name.lower(),
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "204": {"description": schema_name + " deleted successfully"},
                        "404": {"description": schema_name + " not found"}
                    }
                }
            }

            # Create schema for the entity
            schemas[schema_name] = self._create_schema_for_table(table)

        return {
            "openapi": "3.0.0",
            "info": {
                "title": domain + " Data API",
                "version": "1.0.0",
                "description": "RESTful API for " + domain.lower() + " data management and operations"
            },
            "servers": [
                {"url": "https://api." + domain.lower().replace(' ', '') + ".com/v1",
                 "description": "Production " + domain + " API server"}
            ],
            "paths": paths,
            "components": {
                "schemas": schemas,
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            },
            "security": [{"bearerAuth": []}]
        }

    def _create_schema_for_table(self, table_name: str) -> Dict[str, Any]:
        """Create JSON schema for a table"""

        # Base properties all tables have - standard attributes
        properties = {
            "id": {"type": "integer", "description": "Unique identifier"},
            "name": {"type": "string", "description": "Display name"},
            "description": {"type": "string", "description": "Detailed description"},
            "created_at": {"type": "string", "format": "date-time", "description": "Creation timestamp"},
            "updated_at": {"type": "string", "format": "date-time", "description": "Last update timestamp"}
        }

        required = ["name"]

        # Add table-specific properties
        if "user" in table_name.lower():
            properties.update({
                "username": {"type": "string", "description": "Unique username"},
                "email": {"type": "string", "format": "email", "description": "Email address"},
                "first_name": {"type": "string", "description": "First name"},
                "last_name": {"type": "string", "description": "Last name"},
                "status": {"type": "string", "enum": ["active", "inactive"], "description": "User status"}
            })
            required.extend(["username", "email"])
        elif "product" in table_name.lower():
            properties.update({
                "sku": {"type": "string", "description": "Stock keeping unit"},
                "price": {"type": "number", "format": "decimal", "description": "Product price"},
                "weight": {"type": "number", "format": "decimal", "description": "Product weight"},
                "category_id": {"type": "integer", "description": "Product category"},
                "status": {"type": "string", "description": "Product status"}
            })
            required.extend(["sku"])
        elif "inventory" in table_name.lower():
            properties.update({
                "product_id": {"type": "integer", "description": "Reference to product"},
                "location_id": {"type": "integer", "description": "Storage location"},
                "quantity_on_hand": {"type": "integer", "description": "Current stock quantity"},
                "quantity_allocated": {"type": "integer", "description": "Allocated quantity"}
            })
            required.extend(["product_id", "quantity_on_hand"])
        else:
            # Generic properties - we already have the standard ones
            properties.update({
                "status": {"type": "string", "description": "Current status"}
            })

        return {
            "type": "object",
            "required": required,
            "properties": properties
        }

    def _infer_domain_from_requirements(self, requirements: Dict[str, Any]) -> str:
        """Infer domain name from requirements"""

        # Handle None or empty requirements
        if not requirements:
            return "Business Management"

        all_text = ""
        for req_list in requirements.values():
            if isinstance(req_list, list):
                all_text += " ".join(str(req) for req in req_list)

        text_lower = all_text.lower()

        if "inventory" in text_lower or "warehouse" in text_lower:
            return "Inventory Management"
        elif "ecommerce" in text_lower or "e-commerce" in text_lower or "shop" in text_lower:
            return "E-commerce"
        elif "customer" in text_lower and "relationship" in text_lower:
            return "CRM"
        elif "patient" in text_lower or "medical" in text_lower:
            return "Healthcare"
        elif "student" in text_lower or "education" in text_lower:
            return "Education"
        elif "financial" in text_lower or "bank" in text_lower:
            return "Financial Services"
        else:
            return "Business Management"

    def _create_data_catalog(self, tables: List[str], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create Collibra data catalog configuration"""

        # Handle None requirements and always set domain
        if not requirements:
            requirements = {}
        
        # Always infer domain from requirements (or use default if empty)
        domain = self._infer_domain_from_requirements(requirements)

        data_assets = []
        for table in tables:
            data_assets.append({
                "name": table.title().replace('_', ' ') + " Data",
                "description": "Master data for " + table.replace('_', ' '),
                "type": "Table",
                "domain": domain,
                "steward": "Data Steward",
                "sensitivity": "Internal",
                "retention_period": "7 years"
            })

        return {
            "data_assets": data_assets,
            "business_terms": [
                {
                    "name": "Data Quality",
                    "definition": "Measure of data accuracy, completeness, and consistency",
                    "context": domain + " data management"
                },
                {
                    "name": "Master Data",
                    "definition": "Core business entities that are shared across systems",
                    "context": "Reference data for " + domain.lower()
                }
            ],
            "data_quality_rules": [
                {
                    "name": "Primary Key Uniqueness",
                    "description": "All primary keys must be unique across the system",
                    "rule_type": "Uniqueness",
                    "severity": "Critical"
                },
                {
                    "name": "Required Field Completeness",
                    "description": "All required fields must have valid values",
                    "rule_type": "Completeness",
                    "severity": "High"
                }
            ]
        }

    def _create_unity_catalog(self, tables: List[str]) -> Dict[str, Any]:
        """Create Databricks Unity Catalog configuration"""

        return {
            "catalog_name": "business_data_catalog",
            "description": "Unity Catalog for business data management",
            "schemas": [
                {
                    "schema_name": "core_data",
                    "description": "Core business entities and master data",
                    "tables": tables
                },
                {
                    "schema_name": "analytics",
                    "description": "Analytics and reporting views",
                    "tables": [table + "_analytics" for table in tables[:3]]
                }
            ],
            "permissions": {
                "data_engineers": ["SELECT", "INSERT", "UPDATE", "DELETE"],
                "data_analysts": ["SELECT"],
                "business_users": ["SELECT"],
                "administrators": ["ALL"]
            }
        }

    def _create_data_lineage(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create data lineage documentation"""

        return {
            "source_systems": [
                {
                    "name": "Application Database",
                    "type": "PostgreSQL",
                    "description": "Primary operational database"
                },
                {
                    "name": "External APIs",
                    "type": "REST APIs",
                    "description": "Third-party data sources"
                }
            ],
            "transformations": [
                {
                    "name": "Data Validation",
                    "description": "Validate data quality and business rules",
                    "type": "Data Quality"
                },
                {
                    "name": "Data Aggregation",
                    "description": "Aggregate data for reporting and analytics",
                    "type": "Analytics"
                }
            ],
            "target_systems": [
                {
                    "name": "Data Warehouse",
                    "type": "Analytics Platform",
                    "description": "Historical data for reporting and analytics"
                },
                {
                    "name": "Business Intelligence",
                    "type": "BI Platform",
                    "description": "Dashboards and reports for business users"
                }
            ]
        }

    def _create_api_documentation(self, api_spec: Dict[str, Any]) -> str:
        """Create API documentation"""

        title = api_spec.get("info", {}).get("title", "Data API")
        version = api_spec.get("info", {}).get("version", "1.0.0")
        description = api_spec.get("info", {}).get("description", "RESTful API for data management")

        doc_parts = [
            f"# {title} Documentation",
            "",
            f"**Version:** {version}",
            f"**Description:** {description}",
            "",
            "## Overview",
            "This API provides comprehensive access to business data through RESTful endpoints.",
            "",
            "## Authentication",
            "All API endpoints require bearer token authentication.",
            "Include your API token in the Authorization header:",
            "```",
            "Authorization: Bearer <your-api-token>",
            "```",
            "",
            "## Endpoints",
            "The API supports standard CRUD operations for all major business entities:",
            "",
            "- **GET** endpoints for listing and retrieving data",
            "- **POST** endpoints for creating new records",
            "- **PUT** endpoints for updating existing records",
            "- **DELETE** endpoints for removing records",
            "",
            "## Pagination",
            "List endpoints support pagination using `limit` and `offset` parameters:",
            "- `limit`: Number of records to return (default: 50, max: 100)",
            "- `offset`: Number of records to skip (default: 0)",
            "",
            "## Error Handling",
            "The API uses standard HTTP status codes:",
            "- `200` - Success",
            "- `201` - Created",
            "- `400` - Bad Request",
            "- `401` - Unauthorized",
            "- `404` - Not Found",
            "- `500` - Internal Server Error",
            "",
            "## Rate Limiting",
            "API requests are limited to 1000 calls per hour per API key.",
            "",
            "## Support",
            "For technical support, contact the development team."
        ]

        return "\n".join(doc_parts)

    def _create_governance_policies(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create data governance policies"""

        return [
            {
                "policy_name": "Data Access Control",
                "description": "Controls access to sensitive data based on user roles and permissions",
                "policy_type": "Access Control",
                "scope": "All Data Assets",
                "rules": [
                    "Users can only access data relevant to their role",
                    "Sensitive data requires additional authorization",
                    "All data access is logged for audit purposes"
                ]
            },
            {
                "policy_name": "Data Retention and Archival",
                "description": "Defines retention periods and archival procedures for different data types",
                "policy_type": "Data Lifecycle",
                "scope": "All Data Assets",
                "rules": [
                    "Operational data retained for 7 years",
                    "Analytics data retained for 5 years",
                    "Personal data follows applicable privacy regulations"
                ]
            },
            {
                "policy_name": "Data Quality Standards",
                "description": "Ensures data quality and consistency across all systems",
                "policy_type": "Data Quality",
                "scope": "All Data Assets",
                "rules": [
                    "All data must pass quality validation checks",
                    "Data anomalies must be investigated and resolved",
                    "Quality metrics must be monitored and reported"
                ]
            },
            {
                "policy_name": "Data Privacy and Security",
                "description": "Protects sensitive and personal data in compliance with regulations",
                "policy_type": "Privacy and Security",
                "scope": "Sensitive Data",
                "rules": [
                    "Personal data must be encrypted at rest and in transit",
                    "Data access requires user authentication and authorization",
                    "Data breaches must be reported within 24 hours"
                ]
            }
        ]

# ============================================================================
# ONTOLOGY AGENT
# ============================================================================

class OntologyAgent:
    """Creates ontology models in OWL and JSON-LD formats"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """Create ontology models from requirements and conceptual model"""
        print("🔄 Creating Ontology Models...")

        requirements = state.get("requirements")
        conceptual_model = state.get("conceptual_model")
        glossary = state.get("glossary")

        if not requirements or not conceptual_model:
            state["errors"].append("No requirements or conceptual model available for ontology creation")
            return state

        # Generate ontology artifacts
        owl_ontology = self._generate_owl_ontology(requirements, conceptual_model, glossary)
        jsonld_context = self._generate_jsonld_context(requirements, conceptual_model, glossary)

        # Save to files
        if self.artifact_manager:
            ontology_artifacts = {
                "owl_ontology": owl_ontology,
                "jsonld_context": jsonld_context
            }
            self.artifact_manager.save_ontology_artifacts(ontology_artifacts)

        state["ontology_artifacts"] = ontology_artifacts
        state["current_step"] = "ontology_complete"
        return state

    def _generate_owl_ontology(self, requirements, conceptual_model, glossary=None):
        """Generate OWL ontology representation"""

        # Extract entities and relationships from conceptual model
        entities = self._extract_entities_from_conceptual(conceptual_model)
        relationships = self._extract_relationships_from_conceptual(conceptual_model)

        # Get domain context
        domain = self._infer_domain_from_requirements(requirements)

        # Generate OWL using LLM
        if self.llm:
            prompt_parts = [
                "Create an OWL (Web Ontology Language) ontology based on the following domain information:",
                "",
                f"Domain: {domain}",
                "",
                "Entities:",
                "\n".join([f"- {entity}" for entity in entities]),
                "",
                "Relationships:",
                "\n".join([f"- {rel}" for rel in relationships]),
                "",
                "Requirements:",
                json.dumps(requirements, indent=2),
                "",
                "Instructions:",
                "1. Create a comprehensive OWL ontology with classes for each entity",
                "2. Define object properties for relationships between entities",
                "3. Include data properties for entity attributes",
                "4. Define appropriate domain and range constraints",
                "5. Add annotations for human-readable descriptions",
                "6. Use standard OWL/RDF syntax with XML format",
                "",
                "Output the complete OWL ontology in RDF/XML format."
            ]

            prompt = "\n".join(prompt_parts)

            try:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Extract OWL XML from response
                owl_content = self._extract_owl_from_response(response_content)

                # Validate
                if self._validate_owl(owl_content):
                    return owl_content

            except Exception as e:
                print(f"   ⚠️ LLM ontology generation failed: {e}")
                # Continue with fallback

        # Fallback: Create basic OWL ontology
        return self._create_fallback_owl_ontology(entities, relationships, domain)

    def _generate_jsonld_context(self, requirements, conceptual_model, glossary=None):
        """Generate JSON-LD context for the domain"""

        # Extract entities and attributes from conceptual model
        entities = self._extract_entities_from_conceptual(conceptual_model)

        # Get domain context
        domain = self._infer_domain_from_requirements(requirements)
        domain_prefix = domain.lower().replace(" ", "")

        # Create basic JSON-LD context
        context = {
            "@context": {
                domain_prefix: f"http://{domain_prefix}.org/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            }
        }

        # Add entity mappings
        for entity in entities:
            entity_key = entity.lower()
            context["@context"][entity_key] = {
                "@id": f"{domain_prefix}:{entity}"
            }

            # Add basic properties
            context["@context"][f"{entity_key}Id"] = {
                "@id": f"{domain_prefix}:{entity}Id",
                "@type": "xsd:string"
            }
            context["@context"][f"has{entity}"] = {
                "@id": f"{domain_prefix}:has{entity}",
                "@type": f"@id"
            }

        # If using LLM, enhance the context
        if self.llm:
            prompt_parts = [
                "Create a JSON-LD context for the following domain:",
                "",
                f"Domain: {domain}",
                "",
                "Entities:",
                "\n".join([f"- {entity}" for entity in entities]),
                "",
                "Requirements:",
                json.dumps(requirements, indent=2),
                "",
                "Instructions:",
                "1. Create a comprehensive JSON-LD @context object",
                "2. Define appropriate term mappings for entities and properties",
                "3. Include type information for properties where applicable",
                "4. Use standard prefixes for well-known vocabularies",
                "",
                "Return the complete JSON-LD context as a JSON object."
            ]

            prompt = "\n".join(prompt_parts)

            try:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)

                # Extract JSON-LD from response
                jsonld_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if jsonld_match:
                    try:
                        enhanced_context = json.loads(jsonld_match.group())
                        if "@context" in enhanced_context:
                            return enhanced_context
                    except:
                        pass

            except Exception as e:
                print(f"   ⚠️ LLM JSON-LD generation failed: {e}")
                # Continue with basic context

        return context

    def _extract_entities_from_conceptual(self, conceptual_model):
        """Extract entity names from conceptual model"""
        entities = []

        try:
            root = ET.fromstring(conceptual_model)
            for entity in root.findall(".//entity"):
                name = entity.get("name")
                if name:
                    entities.append(name)
        except:
            # Fallback: extract from text
            entity_matches = re.findall(r'<entity name="([^"]+)"', conceptual_model)
            entities = [match for match in entity_matches]

        return entities or ["MainEntity", "SecondaryEntity", "Configuration"]

    def _extract_relationships_from_conceptual(self, conceptual_model):
        """Extract relationships from conceptual model"""
        relationships = []

        try:
            root = ET.fromstring(conceptual_model)
            for relationship in root.findall(".//relationship"):
                name = relationship.get("name")
                from_entity = relationship.get("from")
                to_entity = relationship.get("to")
                cardinality = relationship.get("cardinality")

                if name and from_entity and to_entity:
                    rel_str = f"{name}: {from_entity} → {to_entity}"
                    if cardinality:
                        rel_str += f" ({cardinality})"
                    relationships.append(rel_str)
        except:
            # Fallback: extract from text
            rel_matches = re.findall(r'<relationship name="([^"]+)" from="([^"]+)" to="([^"]+)"', conceptual_model)
            relationships = [f"{name}: {from_entity} → {to_entity}" for name, from_entity, to_entity in rel_matches]

        return relationships

    def _infer_domain_from_requirements(self, requirements):
        """Infer domain name from requirements"""

        # Handle None or empty requirements
        if not requirements:
            return "Business Management"

        all_text = ""
        for req_list in requirements.values():
            if isinstance(req_list, list):
                all_text += " ".join(str(req) for req in req_list)

        text_lower = all_text.lower()

        if "inventory" in text_lower or "warehouse" in text_lower:
            return "Inventory Management"
        elif "ecommerce" in text_lower or "e-commerce" in text_lower or "shop" in text_lower:
            return "E-commerce"
        elif "customer" in text_lower and "relationship" in text_lower:
            return "CRM"
        elif "patient" in text_lower or "medical" in text_lower:
            return "Healthcare"
        elif "student" in text_lower or "education" in text_lower:
            return "Education"
        elif "financial" in text_lower or "bank" in text_lower:
            return "Financial Services"
        else:
            return "Business Management"

    def _extract_owl_from_response(self, response):
        """Extract OWL XML from LLM response"""
        if "<?xml" in response:
            start = response.find("<?xml")
            end_markers = ["</rdf:RDF>", "</Ontology>"]
            end = -1
            for marker in end_markers:
                marker_pos = response.find(marker, start)
                if marker_pos != -1:
                    end = marker_pos + len(marker)
                    break

            if end != -1:
                return response[start:end]

        return response

    def _validate_owl(self, owl_content):
        """Basic validation of OWL content"""
        try:
            root = ET.fromstring(owl_content)
            has_classes = bool(root.findall(".//*[@rdf:about]") or root.findall(".//*[@rdf:ID]"))
            return has_classes
        except ET.ParseError:
            return False

    def _create_fallback_owl_ontology(self, entities, relationships, domain):
        """Create a basic OWL ontology as fallback"""
        domain_prefix = domain.lower().replace(" ", "")
        ns = f"http://{domain_prefix}.org/ontology#"

        owl_parts = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns="' + ns + '"',
            '     xml:base="' + ns + '"',
            '     xmlns:owl="http://www.w3.org/2002/07/owl#"',
            '     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '     xmlns:xsd="http://www.w3.org/2001/XMLSchema#">',
            '',
            '    <owl:Ontology rdf:about="' + ns + '">',
            '        <rdfs:label>' + domain + ' Ontology</rdfs:label>',
            '        <rdfs:comment>Ontology for the ' + domain + ' domain</rdfs:comment>',
            '    </owl:Ontology>',
            ''
        ]

        # Add classes for entities
        for entity in entities:
            owl_parts.extend([
                '    <!-- Class: ' + entity + ' -->',
                '    <owl:Class rdf:about="' + ns + entity + '">',
                '        <rdfs:label>' + entity + '</rdfs:label>',
                '        <rdfs:comment>The ' + entity + ' class in the ' + domain + ' domain</rdfs:comment>',
                '    </owl:Class>',
                ''
            ])

        # Add object properties for relationships
        for i, rel_str in enumerate(relationships):
            parts = rel_str.split(':')
            if len(parts) == 2:
                rel_name = parts[0].strip()
                endpoints = parts[1].strip().split('→')
                if len(endpoints) == 2:
                    from_entity = endpoints[0].strip()
                    to_entity = endpoints[1].split('(')[0].strip()

                    owl_parts.extend([
                        '    <!-- ObjectProperty: ' + rel_name + ' -->',
                        '    <owl:ObjectProperty rdf:about="' + ns + rel_name + '">',
                        '        <rdfs:label>' + rel_name + '</rdfs:label>',
                        '        <rdfs:domain rdf:resource="' + ns + from_entity + '"/>',
                        '        <rdfs:range rdf:resource="' + ns + to_entity + '"/>',
                        '    </owl:ObjectProperty>',
                        ''
                    ])

        # Add generic data properties
        owl_parts.extend([
            '    <!-- DataProperty: hasName -->',
            '    <owl:DatatypeProperty rdf:about="' + ns + 'hasName">',
            '        <rdfs:label>hasName</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '',
            '    <!-- DataProperty: hasDescription -->',
            '    <owl:DatatypeProperty rdf:about="' + ns + 'hasDescription">',
            '        <rdfs:label>hasDescription</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            '',
            '    <!-- DataProperty: hasIdentifier -->',
            '    <owl:DatatypeProperty rdf:about="' + ns + 'hasIdentifier">',
            '        <rdfs:label>hasIdentifier</rdfs:label>',
            '        <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>',
            '    </owl:DatatypeProperty>',
            ''
        ])

        # Close the RDF tag
        owl_parts.append('</rdf:RDF>')

        return '\n'.join(owl_parts)

# ============================================================================
# REVERSE ENGINEERING AGENT
# ============================================================================

class SimplifiedDDLReverseEngineeringAgent:
    """Simplified reverse engineering agent using the simple parser"""

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.parser = SimpleDDLParser()

    def execute(self, state):
        """Reverse engineer models from DDL"""
        print("🔄 Reverse Engineering Database Schema...")

        ddl_content = state.get("ddl_content")
        if not ddl_content:
            state["errors"].append("No DDL content provided")
            return state

        try:
            # Parse DDL using simple parser
            tables = self.parser.parse_ddl(ddl_content)

            if not tables:
                state["errors"].append("No tables found in DDL")
                return state

            print(f"   Found {len(tables)} tables: {', '.join(t.name for t in tables)}")

            # Generate conceptual model
            conceptual_model = self._generate_conceptual_model_from_tables(tables)
            conceptual_dbml = self._generate_dbml_from_tables(tables, "Conceptual")

            # Generate logical model
            logical_model = self._generate_logical_model_from_tables(tables)
            logical_dbml = self._generate_dbml_from_tables(tables, "Logical")

            # Generate physical model (already have it from DDL)
            physical_model = self._generate_physical_model_from_ddl(tables, ddl_content)

            # Extract implied requirements
            requirements = self._extract_requirements_from_tables(tables)

            # Save all artifacts
            if self.artifact_manager:
                self.artifact_manager.save_model(conceptual_model, "conceptual")
                self.artifact_manager.save_dbml(conceptual_dbml, "conceptual")
                self.artifact_manager.save_model(logical_model, "logical")
                self.artifact_manager.save_dbml(logical_dbml, "logical")
                self.artifact_manager.save_model(physical_model, "physical")
                self.artifact_manager.save_requirements(requirements)

                # Save original DDL
                impl_artifacts = {"sql_ddl": ddl_content}
                self.artifact_manager.save_implementation_artifacts(impl_artifacts)

            # Update state
            state["physical_model"] = physical_model
            state["logical_model"] = logical_model
            state["conceptual_model"] = conceptual_model
            state["conceptual_model_dbml"] = conceptual_dbml
            state["logical_model_dbml"] = logical_dbml
            state["requirements"] = requirements
            state["implementation_artifacts"] = {"sql_ddl": ddl_content}
            state["reverse_engineered"] = True
            state["current_step"] = "reverse_engineering_complete"

            print("   ✅ Reverse engineering completed successfully!")

            return state

        except Exception as e:
            print(f"   ❌ Error during reverse engineering: {e}")
            state["errors"].append(str(e))
            return state

    def _generate_conceptual_model_from_tables(self, tables: List[SimpleTableInfo]) -> str:
        """Generate conceptual model XML from tables"""

        # Convert table names to entity names
        entities = []
        for table in tables:
            entity_name = self._table_to_entity_name(table.name)
            entities.append(entity_name)

        # Build XMI
        xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="ConceptualModel" type="Conceptual" source="reverse_engineered">',
            '    <entities>'
        ]

        for table in tables:
            entity_name = self._table_to_entity_name(table.name)
            xmi_parts.extend([
                f'      <entity name="{entity_name}" description="Business entity for {table.name}">',
                f'        <attribute name="id" type="identifier"/>',
                f'        <attribute name="name" type="string"/>',
                f'        <attribute name="description" type="text"/>',
                f'        <attribute name="created_at" type="datetime"/>',
                f'        <attribute name="updated_at" type="datetime"/>'
            ])

            # Add specific attributes based on columns
            for col in table.columns:
                if col['name'].lower() not in ['id', 'name', 'description', 'created_at', 'updated_at']:
                    attr_type = self._map_sql_type_to_conceptual(col['type'])
                    xmi_parts.append(f'        <attribute name="{col["name"]}" type="{attr_type}"/>')

            xmi_parts.append(f'      </entity>')

        xmi_parts.extend([
            '    </entities>',
            '    <relationships>',
            '      <!-- Relationships will be inferred from foreign keys -->',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(xmi_parts)

    def _generate_dbml_from_tables(self, tables: List[SimpleTableInfo], model_type: str = "Logical") -> str:
        """Generate DBML representation from tables"""

        dbml_parts = [
            f"// {model_type} Data Model in DBML format",
            f"// Generated from DDL reverse engineering",
            ""
        ]

        for table in tables:
            dbml_parts.append(f"Table {table.name} {{")

            for column in table.columns:
                col_type = self._map_sql_type_to_dbml(column['type'])

                constraints = []
                if column.get('is_primary_key'):
                    constraints.append('pk')
                if not column.get('nullable', True):
                    constraints.append('not null')

                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""

                dbml_parts.append(f"  {column['name']} {col_type}{constraint_str}")

            dbml_parts.extend(["}", ""])

        return '\n'.join(dbml_parts)

    def _table_to_entity_name(self, table_name: str) -> str:
        """Convert table name to entity name"""
        # Remove common plural endings
        name = table_name.lower()
        if name.endswith('ies'):
            name = name[:-3] + 'y'
        elif name.endswith('s') and not name.endswith('ss'):
            name = name[:-1]

        # Capitalize first letter of each word
        return ''.join(word.capitalize() for word in name.split('_'))

    def _map_sql_type_to_conceptual(self, sql_type: str) -> str:
        """Map SQL types to conceptual types"""
        sql_type_lower = sql_type.lower()

        if any(t in sql_type_lower for t in ['varchar', 'text', 'char']):
            return 'string'
        elif any(t in sql_type_lower for t in ['int', 'serial']):
            return 'integer'
        elif any(t in sql_type_lower for t in ['decimal', 'numeric', 'float']):
            return 'number'
        elif 'bool' in sql_type_lower:
            return 'boolean'
        elif any(t in sql_type_lower for t in ['date', 'time']):
            return 'datetime'
        else:
            return 'string'

    def _map_sql_type_to_dbml(self, sql_type: str) -> str:
        """Map SQL types to DBML types"""
        sql_type_lower = sql_type.lower()

        if 'serial' in sql_type_lower:
            return 'integer'
        elif 'varchar' in sql_type_lower or 'text' in sql_type_lower:
            return 'varchar'
        elif 'int' in sql_type_lower:
            return 'integer'
        elif 'decimal' in sql_type_lower or 'numeric' in sql_type_lower:
            return 'decimal'
        elif 'timestamp' in sql_type_lower or 'datetime' in sql_type_lower:
            return 'timestamp'
        elif 'date' in sql_type_lower:
            return 'date'
        elif 'bool' in sql_type_lower:
            return 'boolean'
        else:
            return 'varchar'

    def _generate_logical_model_from_tables(self, tables: List[SimpleTableInfo]) -> str:
        """Generate logical model from tables"""

        logical_xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="LogicalModel" type="Logical" source="reverse_engineered">',
            '    <tables>'
        ]

        for table in tables:
            logical_xmi_parts.append(f'      <table name="{table.name}">')

            for column in table.columns:
                nullable = 'true' if column.get('nullable', True) else 'false'
                pk = 'true' if column.get('is_primary_key') else 'false'

                logical_xmi_parts.append(
                    f'        <column name="{column["name"]}" '
                    f'type="{column["type"]}" nullable="{nullable}" '
                    f'primary_key="{pk}"/>'
                )

            logical_xmi_parts.append(f'      </table>')

        logical_xmi_parts.extend([
            '    </tables>',
            '    <relationships>',
            '      <!-- Foreign key relationships detected from DDL -->',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(logical_xmi_parts)

    def _generate_physical_model_from_ddl(self, tables: List[SimpleTableInfo], ddl_content: str) -> str:
        """Generate physical model XMI from DDL"""

        table_count = len(tables)

        xmi_template = '''<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">
  <model name="PhysicalModel" type="Physical" source="reverse_engineered">
    <description>Physical model reverse engineered from SQL DDL ({table_count} tables)</description>
    <sql_ddl><![CDATA[
{sql_ddl}
    ]]></sql_ddl>
  </model>
</xmi:XMI>'''

        return xmi_template.format(
            table_count=table_count,
            sql_ddl=ddl_content
        )

    # def _extract_requirements_from_tables(self, tables: List[SimpleTableInfo]) -> Dict[str, Any]:
    #     """Extract implied requirements from the schema"""

    #     requirements = {
    #         "functional_requirements": [],
    #         "non_functional_requirements": [],
    #         "data_requirements": [],
    #         "integration_requirements": [],
    #         "compliance_requirements": []
    #     }

    def _extract_requirements_from_tables(self, tables: List[SimpleTableInfo]) -> Dict[str, Any]:
        """Extract implied requirements from the schema"""

        requirements = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "compliance_requirements": [],
            "inferred_entities": []  # Add this
        }

        # Extract entity names properly
        entity_names = []
        for table in tables:
            entity = self._table_to_entity_name(table.name)
            entity_names.append(entity)
            requirements["functional_requirements"].append(f"Manage {entity} data")

        requirements["inferred_entities"] = entity_names

        # Rest of the method continues...

        # Analyze tables for patterns
        has_timestamps = False
        has_status_fields = False

        for table in tables:
            entity = self._table_to_entity_name(table.name)
            requirements["functional_requirements"].append(f"Manage {entity} data")

            for col in table.columns:
                col_name_lower = col['name'].lower()
                if 'created' in col_name_lower or 'updated' in col_name_lower:
                    has_timestamps = True
                if 'status' in col_name_lower or 'active' in col_name_lower:
                    has_status_fields = True

        # Add implied requirements based on patterns
        if has_timestamps:
            requirements["compliance_requirements"].append("Audit trail for data changes")
        if has_status_fields:
            requirements["functional_requirements"].append("Status management capability")

        # Add standard requirements
        requirements["non_functional_requirements"].extend([
            "Maintain referential integrity",
            "Support concurrent access",
            "Ensure data consistency"
        ])

        requirements["data_requirements"] = [
            f"Store and manage {len(tables)} entity types",
            "Support relationships between entities"
        ]

        return requirements

class EnhancedImplementationAgent:
    """Enhanced implementation agent that provisions MySQL, SQL Server, and SQLite databases"""

    def __init__(self, llm=None, artifact_manager=None, db_config=None):
        """
        Initialize the enhanced implementation agent

        Args:
            llm: Language model for generating sample data
            artifact_manager: Artifact manager for saving files
            db_config: Database configuration (host, user, password, etc.)
        """
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.db_config = db_config or self._get_default_db_config()
        self.faker = Faker()
        self.connection = None
        self.database_name = None
        self.use_sqlite_fallback = False
        self.database_type = None

    def _get_default_db_config(self):
        """Get default database configuration with SQL Server support"""
        return {
            'preferred_type': 'sqlserver',  # Can be 'sqlserver', 'mysql', or 'sqlite'

            # SQL Server / Azure SQL configuration
            'sqlserver': {
                'server': os.getenv('AZURE_SQL_SERVER', 'your-server.database.windows.net'),
                'database': None,  # Will be set based on project
                'username': os.getenv('AZURE_SQL_USERNAME', ''),
                'password': os.getenv('AZURE_SQL_PASSWORD', ''),
                'driver': '{ODBC Driver 17 for SQL Server}',
                'port': 1433,
                'encrypt': True,
                'trust_server_certificate': False,
                'connection_timeout': 30,
                'authentication': 'SqlPassword'
            },

            # MySQL configuration (existing)
            'mysql': {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'port': 3306,
                'raise_on_warnings': True
            },

            # SQLite configuration (existing)
            'sqlite': {
                'use_as_fallback': True
            }
        }

    def execute(self, state):
        """Execute the enhanced implementation process with SQL Server support"""
        print("🗄️ Enhanced Implementation Agent Starting...")

        # Check if data was already loaded by DDL+CSV reverse engineering
        data_already_loaded = (
            state.get("data_loaded") or
            state.get("current_step") == "ddl_csv_reverse_engineering_complete" or
            state.get("reverse_engineered_with_data")
        )

        if data_already_loaded:
            print("   ℹ️ Data was already loaded from CSV files during reverse engineering.")
            print("   ✅ Skipping synthetic data generation to preserve real data.")

            # Get database info from state
            impl_artifacts = state.get("implementation_artifacts", {})
            db_info = impl_artifacts.get("database_info", {})

            if db_info:
                state["database_path"] = db_info.get("database_file")  # ADD THIS
                # Generate access instructions without overwriting data
                access_info = self._generate_access_instructions(db_info)
                connection_script = self._generate_connection_script(db_info)

                # Update implementation artifacts without sample data
                updated_artifacts = {
                    "database_info": db_info,
                    "access_instructions": access_info,
                    "connection_script": connection_script,
                    "data_source": "CSV files (reverse engineered - real data)"
                }

                # Save artifacts if artifact manager available
                if self.artifact_manager:
                    self._save_implementation_artifacts(updated_artifacts)

                state["implementation_artifacts"] = updated_artifacts
                state["database_provisioned"] = True
                state["database_path"] = db_info.get("database_file")  # ADD THIS LINE
                state["current_step"] = "implementation_complete"

                # Display access instructions
                print("\n" + "="*60)
                print("📌 DATABASE ACCESS INSTRUCTIONS")
                print("="*60)
                print(access_info)

                print("\n   ✅ Implementation complete - preserved existing CSV data")

                return state

        # Check if this is a regular CSV reverse engineering case (without DDL)
        is_csv_reverse = bool(state.get("inference_result"))

        if state.get("data_already_loaded") and is_csv_reverse:
            # This is the existing CSV inference path (without DDL)
            print("   ℹ️ Data was loaded by DataInferenceAgent.")
            return state

        # Original code for cases that need synthetic data generation OR Lab workflow CSV loading
        print("   ℹ️ Processing database provisioning...")

        # Get SQL DDL from state
        sql_ddl = state.get("sql_ddl") or state.get("implementation_artifacts", {}).get("sql_ddl")

        if not sql_ddl:
            state["errors"].append("No SQL DDL available for implementation")
            return state

        # Generate database name from project
        project_id = state.get("project_id", "data_architect_db")
        self.database_name = f"{project_id}_db".replace("-", "_").replace(" ", "_").lower()

        # Check if CSV files were provided (Lab workflow)
        csv_files = state.get("csv_files")
        has_csv_data = csv_files and isinstance(csv_files, list) and len(csv_files) > 0

        try:
            # Step 1: Provision database
            print("   1️⃣ Provisioning database...")
            db_info = self._provision_database(sql_ddl)

            # Step 2: Generate and load sample data OR load CSV data
            if has_csv_data:
                print(f"   2️⃣ Loading provided CSV data ({len(csv_files)} files)...")
                load_results = self._load_csv_files(csv_files)
            else:
                print("   2️⃣ Generating sample data...")
                sample_data = self._generate_sample_data(sql_ddl, state.get("requirements"))

                print("   3️⃣ Loading sample data...")
                load_results = self._load_sample_data(sample_data)

            # Step 4: Generate access instructions
            print("   4️⃣ Generating access instructions...")
            access_info = self._generate_access_instructions(db_info)

            # Save artifacts
            implementation_artifacts = {
                "database_info": db_info,
                "load_results": load_results,
                "access_instructions": access_info,
                "connection_script": self._generate_connection_script(db_info),
                "data_source": "CSV files (Lab workflow)" if has_csv_data else "Synthetic data (generated)"
            }

            if not has_csv_data:
                implementation_artifacts["sample_data"] = sample_data

            # Save to files if artifact manager available
            if self.artifact_manager:
                self._save_implementation_artifacts(implementation_artifacts)

            # Update state
            state["implementation_artifacts"] = implementation_artifacts
            state["database_provisioned"] = True
            state["database_path"] = implementation_artifacts["database_info"].get("database_file")  # ADD THIS
            state["current_step"] = "implementation_complete"

            print("   ✅ Implementation completed successfully!")
            print(f"   📊 Database: {db_info['database_name']}")
            print(f"   📋 Tables created: {db_info['tables_created']}")
            print(f"   📝 Sample records loaded: {load_results.get('total_records', 0)}")

            # Display access instructions
            print("\n" + "="*60)
            print("📌 DATABASE ACCESS INSTRUCTIONS")
            print("="*60)
            print(access_info)

            return state

        except Exception as e:
            print(f"   ❌ Implementation error: {e}")
            state["errors"].append(f"Implementation failed: {str(e)}")
            return state
        finally:
            self._close_connection()

    def _load_csv_files(self, csv_file_paths: List[str]) -> Dict[str, Any]:
        """Load data from CSV file paths"""
        import pandas as pd

        load_results = {
            "tables_loaded": [],
            "total_records": 0,
            "errors": []
        }

        cursor = self.connection.cursor()

        for csv_path in csv_file_paths:
            try:
                # Get table name from filename
                table_name = Path(csv_path).stem.upper()

                print(f"      Processing {Path(csv_path).name} -> {table_name}")

                # Check if table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND UPPER(name)=?",
                    (table_name,)
                )
                table_exists = cursor.fetchone()

                if not table_exists:
                    print(f"      ⚠️ Warning: Table {table_name} not found in database")
                    load_results["errors"].append(f"Table {table_name} not found")
                    continue

                # Get actual table name (preserve case)
                actual_table_name = table_exists[0]

                # Read CSV
                df = pd.read_csv(csv_path)
                print(f"      Read {len(df)} rows from CSV")

                # Get table schema
                cursor.execute(f"PRAGMA table_info({actual_table_name})")
                table_columns = [row[1] for row in cursor.fetchall()]

                print(f"      Table expects columns: {table_columns}")
                print(f"      CSV has columns: {list(df.columns)}")

                # Map CSV columns to table columns (case-insensitive)
                column_mapping = {}
                for table_col in table_columns:
                    for csv_col in df.columns:
                        if csv_col.lower() == table_col.lower():
                            column_mapping[table_col] = csv_col
                            break

                print(f"      Column mapping: {column_mapping}")

                # Create mapped DataFrame
                mapped_data = {}
                for table_col in table_columns:
                    if table_col in column_mapping:
                        mapped_data[table_col] = df[column_mapping[table_col]]
                    else:
                        # Fill with NULL for missing columns
                        print(f"      ⚠️ Column {table_col} not found in CSV, filling with NULL")
                        mapped_data[table_col] = None

                mapped_df = pd.DataFrame(mapped_data)

                # Load into database
                mapped_df.to_sql(actual_table_name, self.connection, if_exists='append', index=False)

                load_results["tables_loaded"].append({
                    "table": actual_table_name,
                    "records": len(mapped_df)
                })
                load_results["total_records"] += len(mapped_df)

                print(f"      ✓ Loaded {len(mapped_df)} records into {actual_table_name}")

            except Exception as e:
                load_results["errors"].append(f"Error loading {csv_path}: {str(e)}")
                print(f"      ✗ Error loading {csv_path}: {e}")
                import traceback
                traceback.print_exc()

        return load_results

    def _provision_database(self, sql_ddl: str) -> Dict[str, Any]:
        """Provision the database based on preferred type"""
        preferred_type = self.db_config.get('preferred_type', 'sqlite')

        # Try preferred database type first
        if preferred_type == 'sqlserver':
            try:
                return self._provision_sqlserver_database(sql_ddl)
            except Exception as sqlserver_error:
                print(f"   ⚠️ SQL Server connection failed: {sqlserver_error}")
                # Try MySQL as fallback
                try:
                    return self._provision_mysql_database(sql_ddl)
                except Exception as mysql_error:
                    print(f"   ⚠️ MySQL connection failed: {mysql_error}")
                    # Finally try SQLite
                    if self.db_config.get('sqlite', {}).get('use_as_fallback', True):
                        print("   🔄 Falling back to SQLite...")
                        return self._provision_sqlite_database(sql_ddl)
                    else:
                        raise sqlserver_error

        elif preferred_type == 'mysql':
            try:
                return self._provision_mysql_database(sql_ddl)
            except Exception as mysql_error:
                print(f"   ⚠️ MySQL connection failed: {mysql_error}")
                # Try SQL Server as fallback
                try:
                    return self._provision_sqlserver_database(sql_ddl)
                except Exception as sqlserver_error:
                    print(f"   ⚠️ SQL Server connection failed: {sqlserver_error}")
                    # Finally try SQLite
                    if self.db_config.get('sqlite', {}).get('use_as_fallback', True):
                        print("   🔄 Falling back to SQLite...")
                        return self._provision_sqlite_database(sql_ddl)
                    else:
                        raise mysql_error

        else:  # Default to SQLite
            return self._provision_sqlite_database(sql_ddl)

    def _provision_sqlite_database(self, sql_ddl: str) -> Dict[str, Any]:
        """Provision SQLite database as fallback"""
        import sqlite3

        self.use_sqlite_fallback = True

        # Create database path
        if self.artifact_manager:
            project_root = self.artifact_manager.project_path.parent
            db_path = project_root / f"{self.database_name}.db"
        else:
            db_path = Path(f"{self.database_name}.db")

        # Remove existing database if exists
        if db_path.exists():
            db_path.unlink()

        print(f"   📁 Creating SQLite database: {db_path}")

        # Create new SQLite database
        self.connection = sqlite3.connect(str(db_path))
        cursor = self.connection.cursor()

        # Convert DDL from PostgreSQL/MySQL to SQLite
        sqlite_ddl = self._convert_ddl_to_sqlite(sql_ddl)

        # Execute DDL statements
        statements = self._split_sql_statements(sqlite_ddl)

        tables_created = 0
        indexes_created = 0

        for statement in statements:
            if statement.strip():
                try:
                    cursor.execute(statement)
                    if 'CREATE TABLE' in statement.upper():
                        tables_created += 1
                    elif 'CREATE INDEX' in statement.upper():
                        indexes_created += 1
                except Exception as e:
                    print(f"   ⚠️ Statement execution warning: {e}")

        self.connection.commit()

        # Get table list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]

        self.database_type = 'sqlite'

        return {
            "database_type": "SQLite",
            "database_name": self.database_name,
            "database_file": str(db_path),
            "tables_created": tables_created,
            "indexes_created": indexes_created,
            "table_list": tables,
            "connection_string": f"sqlite:///{db_path}"
        }

    def _convert_ddl_to_sqlite(self, ddl: str) -> str:
        """Convert PostgreSQL/MySQL DDL to SQLite compatible format"""
        sqlite_ddl = ddl

        # Remove unsupported features
        sqlite_ddl = sqlite_ddl.replace('SERIAL', 'INTEGER')
        sqlite_ddl = sqlite_ddl.replace('AUTO_INCREMENT', 'AUTOINCREMENT')
        sqlite_ddl = sqlite_ddl.replace('DEFAULT CURRENT_TIMESTAMP', "DEFAULT (datetime('now'))")
        sqlite_ddl = sqlite_ddl.replace('GENERATED ALWAYS AS', '--GENERATED ALWAYS AS')
        sqlite_ddl = sqlite_ddl.replace('CREATE EXTENSION', '-- CREATE EXTENSION')

        # Convert data types
        type_mappings = {
            'VARCHAR': 'TEXT',
            'DECIMAL': 'REAL',
            'TIMESTAMP': 'DATETIME',
            'BOOLEAN': 'INTEGER'
        }

        for pg_type, sqlite_type in type_mappings.items():
            sqlite_ddl = sqlite_ddl.replace(f'{pg_type}(', f'{sqlite_type}(')
            sqlite_ddl = sqlite_ddl.replace(f'{pg_type} ', f'{sqlite_type} ')

        return sqlite_ddl

    def _split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements"""
        lines = sql.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('--'):
                cleaned_lines.append(line)

        cleaned_sql = '\n'.join(cleaned_lines)
        statements = cleaned_sql.split(';')
        return [stmt.strip() for stmt in statements if stmt.strip()]

    def _generate_sample_data(self, sql_ddl: str, requirements: Dict = None) -> Dict[str, List[Dict]]:
        """Generate sample data for all tables"""

        sample_data = {}

        # Parse DDL to get table structures
        tables = self._parse_tables_from_ddl(sql_ddl)

        for table_name, columns in tables.items():
            print(f"      Generating data for {table_name}...")

            # Determine number of records based on table type
            num_records = self._determine_record_count(table_name)

            # Generate records
            records = []
            for i in range(num_records):
                record = self._generate_record(table_name, columns, i)
                records.append(record)

            sample_data[table_name] = records

        return sample_data

    def _parse_tables_from_ddl(self, ddl: str) -> Dict[str, List[Dict]]:
        """Parse table structures from DDL"""
        tables = {}

        # Simple regex parsing for CREATE TABLE statements
        import re

        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        matches = re.findall(create_table_pattern, ddl, re.DOTALL | re.IGNORECASE)

        for table_name, columns_text in matches:
            columns = []

            # Parse columns
            column_lines = columns_text.split(',')
            for line in column_lines:
                line = line.strip()
                if line and not any(keyword in line.upper() for keyword in ['CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 'CHECK']):
                    parts = line.split()
                    if len(parts) >= 2:
                        col_name = parts[0]
                        col_type = parts[1]
                        columns.append({
                            'name': col_name,
                            'type': col_type,
                            'is_primary': 'PRIMARY KEY' in line.upper()
                        })

            tables[table_name] = columns

        return tables

    def _determine_record_count(self, table_name: str) -> int:
        """Determine how many sample records to generate"""
        table_lower = table_name.lower()

        # Configuration tables: fewer records
        if any(word in table_lower for word in ['config', 'settings', 'category', 'type', 'status']):
            return random.randint(5, 10)

        # Main entity tables: moderate records
        elif any(word in table_lower for word in ['user', 'customer', 'product', 'employee']):
            return random.randint(20, 50)

        # Transaction tables: more records
        elif any(word in table_lower for word in ['order', 'transaction', 'log', 'history']):
            return random.randint(50, 100)

        # Junction tables: many records
        elif '_' in table_name and any(word in table_lower for word in ['item', 'detail', 'line']):
            return random.randint(100, 200)

        # Default
        else:
            return random.randint(10, 30)

    def _generate_record(self, table_name: str, columns: List[Dict], index: int) -> Dict:
        """Generate a single record with realistic data"""
        record = {}

        for column in columns:
            col_name = column['name']
            col_type = column['type'].upper()

            # Skip auto-increment/serial columns
            if column.get('is_primary') and 'INT' in col_type:
                record[col_name] = index + 1

            # Generate based on column name patterns
            elif col_name.lower() in ['name', 'first_name']:
                record[col_name] = f"name{random.randint(10, 99)}"
            elif col_name.lower() in ['description']:
                record[col_name] = f"description{random.randint(10, 99)}"
            elif col_name.lower() in ['last_name', 'surname']:
                record[col_name] = self.faker.last_name()
            elif col_name.lower() in ['email', 'email_address']:
                record[col_name] = self.faker.email()
            elif col_name.lower() in ['phone', 'phone_number', 'telephone']:
                record[col_name] = self.faker.phone_number()
            elif col_name.lower() in ['address', 'street_address']:
                record[col_name] = self.faker.address().replace('\n', ', ')
            elif col_name.lower() in ['city']:
                record[col_name] = self.faker.city()
            elif col_name.lower() in ['state', 'province']:
                record[col_name] = self.faker.state()
            elif col_name.lower() in ['country']:
                record[col_name] = self.faker.country()
            elif col_name.lower() in ['postal_code', 'zip_code', 'zip']:
                record[col_name] = self.faker.postcode()
            elif col_name.lower() in ['company', 'company_name', 'organization']:
                record[col_name] = self.faker.company()
            elif col_name.lower() in ['job_title', 'position', 'title']:
                record[col_name] = self.faker.job()
            elif col_name.lower() in ['description', 'notes', 'comments']:
                record[col_name] = self.faker.text(max_nb_chars=200)
            elif col_name.lower() in ['sku', 'product_code']:
                record[col_name] = f"SKU-{random.randint(10000, 99999)}"
            elif col_name.lower() in ['username', 'user_name']:
                record[col_name] = self.faker.user_name()
            elif col_name.lower() in ['password', 'password_hash']:
                record[col_name] = self.faker.sha256()
            elif col_name.lower() in ['url', 'website']:
                record[col_name] = self.faker.url()
            elif col_name.lower() in ['status']:
                record[col_name] = random.choice(['active', 'inactive', 'pending', 'completed'])
            elif col_name.lower() in ['payment_method']:
                record[col_name] = random.choice(['cash', 'credit', 'debit', 'PayPal'])
            elif col_name.lower() in ['created_at', 'created_date', 'date_created']:
                record[col_name] = self.faker.date_time_between(start_date='-2y', end_date='now')
            elif col_name.lower() in ['updated_at', 'modified_at', 'last_modified']:
                record[col_name] = self.faker.date_time_between(start_date='-30d', end_date='now')

            # Generate based on data type
            elif 'INT' in col_type or 'SERIAL' in col_type:
                if 'price' in col_name.lower() or 'amount' in col_name.lower():
                    record[col_name] = round(random.uniform(10, 1000), 2)
                elif 'quantity' in col_name.lower() or 'count' in col_name.lower():
                    record[col_name] = random.randint(1, 100)
                elif 'age' in col_name.lower():
                    record[col_name] = random.randint(18, 80)
                elif 'year' in col_name.lower():
                    record[col_name] = random.randint(2015, 2024)
                else:
                    record[col_name] = random.randint(1, 1000)

            elif 'DECIMAL' in col_type or 'FLOAT' in col_type or 'REAL' in col_type:
                record[col_name] = round(random.uniform(0, 1000), 2)

            elif 'DATE' in col_type:
                record[col_name] = self.faker.date_between(start_date='-2y', end_date='today')

            elif 'TIME' in col_type:
                if 'DATE' not in col_type:  # Just TIME, not DATETIME/TIMESTAMP
                    record[col_name] = self.faker.time()
                else:
                    record[col_name] = self.faker.date_time_between(start_date='-1y', end_date='now')

            elif 'BOOL' in col_type:
                record[col_name] = random.choice([True, False])

            elif 'JSON' in col_type:
                record[col_name] = json.dumps({"key": "value", "number": random.randint(1, 100)})

            # Default for text/varchar
            else:
                if col_name not in record:  # Don't overwrite if already set
                    record[col_name] = self.faker.word().capitalize() + " " + str(index + 1)

        return record

    def _load_sample_data(self, sample_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Load sample data into database"""

        if not self.connection:
            raise Exception("No database connection available")

        cursor = self.connection.cursor() if not self.use_sqlite_fallback else self.connection.cursor()

        load_results = {
            "tables_loaded": [],
            "total_records": 0,
            "errors": []
        }

        for table_name, records in sample_data.items():
            if not records:
                continue

            try:
                # Build INSERT statement
                columns = list(records[0].keys())
                placeholders = ', '.join(['?' if self.use_sqlite_fallback else '%s'] * len(columns))
                insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

                # Insert records
                for record in records:
                    values = [self._format_value(record[col]) for col in columns]
                    cursor.execute(insert_sql, values)

                self.connection.commit()

                load_results["tables_loaded"].append({
                    "table": table_name,
                    "records": len(records)
                })
                load_results["total_records"] += len(records)

                print(f"      ✓ Loaded {len(records)} records into {table_name}")

            except Exception as e:
                load_results["errors"].append(f"Error loading {table_name}: {str(e)}")
                print(f"      ✗ Error loading {table_name}: {e}")

        return load_results

    def _format_value(self, value):
        """Format value for database insertion"""
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(value, bool):
            return 1 if value else 0
        elif value is None:
            return None
        else:
            return str(value)

    def _generate_access_instructions(self, db_info: Dict[str, Any]) -> str:
        """Generate database access instructions"""

        if db_info["database_type"] == "SQLite":
            instructions = f"""
DATABASE CONNECTION DETAILS:
======================================
Database File: {db_info['database_file']}
Location: {Path(db_info['database_file']).absolute()}

CONNECTION METHODS:

1. SQLite Command Line:
   sqlite3 {db_info['database_file']}

2. Python (sqlite3):
   import sqlite3
   connection = sqlite3.connect('{db_info['database_file']}')

3. Python (SQLAlchemy):
   from sqlalchemy import create_engine
   engine = create_engine('{db_info['connection_string']}')

4. Python (Pandas):
   import pandas as pd
   import sqlite3
   conn = sqlite3.connect('{db_info['database_file']}')
   df = pd.read_sql_query("SELECT * FROM table_name", conn)

5. DBeaver/DB Browser for SQLite:
   - Open file: {db_info['database_file']}

AVAILABLE TABLES:
{chr(10).join(['   - ' + table for table in db_info['table_list']])}

SAMPLE QUERIES TO TEST:
   SELECT * FROM {db_info['table_list'][0] if db_info['table_list'] else 'table_name'} LIMIT 10;
   SELECT COUNT(*) FROM {db_info['table_list'][0] if db_info['table_list'] else 'table_name'};
"""
        else:
            instructions = "Database connection instructions not available for this database type."

        return instructions

    def _generate_connection_script(self, db_info: Dict[str, Any]) -> str:
        """Generate a Python script for connecting to the database"""

        if db_info["database_type"] == "SQLite":
            script = f'''#!/usr/bin/env python3
"""
Database Connection Script
Generated by Data Architect Agent
Database: {db_info['database_file']}
"""

import sqlite3
import pandas as pd
from pathlib import Path

def connect_to_database():
    """Connect to the SQLite database"""
    try:
        db_path = '{db_info['database_file']}'

        if not Path(db_path).exists():
            print(f"Database file not found: {{db_path}}")
            return None

        connection = sqlite3.connect(db_path)
        print(f"Successfully connected to SQLite database: {{db_path}}")
        return connection

    except Exception as e:
        print(f"Error connecting to SQLite: {{e}}")
        return None

def test_connection():
    """Test the database connection and display sample data"""

    connection = connect_to_database()
    if not connection:
        return

    try:
        cursor = connection.cursor()

        # Show tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("\\nAvailable Tables:")
        for table in tables:
            print(f"   - {{table[0]}}")

        # Show sample data from first table
        if tables:
            table_name = tables[0][0]
            print(f"\\nSample data from '{{table_name}}':")

            query = f"SELECT * FROM {{table_name}} LIMIT 5"
            df = pd.read_sql_query(query, connection)
            print(df)

    except Exception as e:
        print(f"Error executing query: {{e}}")

    finally:
        connection.close()
        print("\\nDatabase connection closed")

if __name__ == "__main__":
    test_connection()
'''
        else:
            script = "# Connection script not available for this database type"

        return script

    def _save_implementation_artifacts(self, artifacts: Dict[str, Any]):
        """Save implementation artifacts to files"""

        if not self.artifact_manager:
            return

        impl_dir = self.artifact_manager.project_path / "03_implementation"

        # Save database info
        db_info_file = impl_dir / "database_info.json"
        with open(db_info_file, 'w', encoding='utf-8') as f:
            # Convert any datetime objects to strings for JSON serialization
            db_info = artifacts['database_info'].copy()
            json.dump(db_info, f, indent=2, default=str)

        # Save sample data ONLY if it exists (not present in CSV reverse engineering)
        if artifacts.get('sample_data'):
            sample_data_file = impl_dir / "sample_data.json"
            with open(sample_data_file, 'w', encoding='utf-8') as f:
                # Convert datetime objects in sample data
                sample_data_serializable = {}
                for table, records in artifacts['sample_data'].items():
                    sample_data_serializable[table] = []
                    for record in records:
                        serializable_record = {}
                        for key, value in record.items():
                            if isinstance(value, (datetime, )):
                                serializable_record[key] = value.isoformat()
                            else:
                                serializable_record[key] = value
                        sample_data_serializable[table].append(serializable_record)
                json.dump(sample_data_serializable, f, indent=2, default=str)

        # Save access instructions
        access_file = impl_dir / "database_access_instructions.txt"
        with open(access_file, 'w', encoding='utf-8') as f:
            f.write(artifacts['access_instructions'])

        # Save connection script
        script_file = impl_dir / "connect_to_database.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(artifacts['connection_script'])

        # Save load results ONLY if they exist
        if artifacts.get('load_results'):
            load_results_file = impl_dir / "load_results.json"
            with open(load_results_file, 'w', encoding='utf-8') as f:
                json.dump(artifacts['load_results'], f, indent=2, default=str)

        print(f"   📁 Implementation artifacts saved to: {impl_dir}")

    def _close_connection(self):
        """Close database connection"""
        if self.connection:
            try:
                if self.use_sqlite_fallback:
                    self.connection.close()
                else:
                    if hasattr(self.connection, 'is_connected') and self.connection.is_connected():
                        self.connection.close()
            except:
                pass

    # MySQL and SQL Server provisioning methods would go here
    # (I've omitted them for brevity since you're using SQLite)
    def _provision_mysql_database(self, sql_ddl: str) -> Dict[str, Any]:
        """Provision MySQL database (placeholder)"""
        raise NotImplementedError("MySQL provisioning not included in this version")

    def _provision_sqlserver_database(self, sql_ddl: str) -> Dict[str, Any]:
        """Provision SQL Server database (placeholder)"""
        raise NotImplementedError("SQL Server provisioning not included in this version")

#==========================
#
# Visualization Agent
#
#==========================


class VisualizationAgent:
    """Creates interactive visualizations and statistical summaries for database tables"""

    def __init__(self, llm=None, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.connection = None
        self.database_path = None

    def execute(self, state):
        """Generate visualizations from the database"""
        print("📊 Creating Data Visualizations...")

        # Find the database
        db_path = self._find_database(state)
        if not db_path:
            state["errors"].append("No database found for visualization")
            return state

        self.database_path = db_path

        try:
            # Connect to database
            import sqlite3
            self.connection = sqlite3.connect(str(db_path))

            # Get table information
            tables_info = self._get_tables_info()

            # Generate statistical summaries
            stats_summary = self._generate_statistics_summary(tables_info)

            # Generate visualizations
            plot_artifacts = self._generate_plots(tables_info)

            # Generate interactive table view
            table_view_html = self._generate_interactive_table_view(tables_info)

            # Generate standalone data explorer
            explorer_html = self._generate_data_explorer(tables_info)

            # Generate statistical report HTML
            stats_html = self._generate_stats_html(stats_summary, plot_artifacts)

            # Generate Jupyter widgets if in notebook environment
            widgets = None
            if self._in_notebook():
                widgets = self._create_jupyter_widgets(tables_info)
                # Display statistics and plots in notebook
                self._display_stats_in_notebook(stats_summary, plot_artifacts)

            # Save artifacts
            visualization_artifacts = {
                "table_view_html": table_view_html,
                "data_explorer_html": explorer_html,
                "stats_html": stats_html,
                "tables_info": tables_info,
                "statistics_summary": stats_summary,
                "plot_artifacts": plot_artifacts
            }

            if self.artifact_manager:
                saved_files = self._save_visualization_artifacts(visualization_artifacts)
                state["visualization_files"] = saved_files

            state["visualization_artifacts"] = visualization_artifacts
            state["visualization_widgets"] = widgets
            state["current_step"] = "visualization_complete"

            print(f"   ✅ Created visualizations for {len(tables_info)} tables")

            # Display widgets if in notebook
            if widgets and self._in_notebook():
                print("\n📊 Interactive Table Viewer:")
                display(widgets)

            return state

        except Exception as e:
            print(f"   ❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
            state["errors"].append(f"Visualization failed: {str(e)}")
            return state
        finally:
            if self.connection:
                self.connection.close()

    def _is_id_column(self, column_name):
        """Check if a column is an ID column that should be excluded from statistics"""
        col_lower = column_name.lower()
        # Exclude columns that are IDs or foreign keys
        return (col_lower == 'id' or
                col_lower.endswith('_id') or
                col_lower.endswith('_code') or  # Often codes are identifiers
                col_lower.endswith('_number') and 'phone' not in col_lower or  # Exclude ID numbers but not phone numbers
                col_lower in ['pk', 'fk', 'key'])

    def _generate_statistics_summary(self, tables_info):
        """Generate statistical summaries for all numeric columns (excluding IDs)"""
        import pandas as pd
        import numpy as np

        stats = {}

        for table_name, info in tables_info.items():
            # Convert to DataFrame
            df = pd.DataFrame(info['sample_rows'], columns=info['column_names'])

            # Get numeric columns, excluding ID columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not self._is_id_column(col)]

            table_stats = {
                'total_rows': info['total_count'],
                'numeric_columns': {},
                'categorical_columns': {}
            }

            # Statistics for numeric columns (excluding IDs)
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    table_stats['numeric_columns'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()) if len(df[col]) > 1 else 0,
                        'q25': float(df[col].quantile(0.25)),
                        'q75': float(df[col].quantile(0.75)),
                        'null_count': int(df[col].isna().sum()),
                        'unique_count': int(df[col].nunique())
                    }

            # Statistics for categorical columns (also excluding ID-like columns)
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if not self._is_id_column(col)]

            for col in categorical_cols:
                value_counts = df[col].value_counts()
                table_stats['categorical_columns'][col] = {
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isna().sum()),
                    'mode': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
                    'top_values': value_counts.head(5).to_dict()
                }

            stats[table_name] = table_stats

        return stats

    def _generate_plots(self, tables_info):
        """Generate various plots for numeric data (excluding ID columns)"""
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend for Flask/threading
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        plots = {}

        for table_name, info in tables_info.items():
            df = pd.DataFrame(info['sample_rows'], columns=info['column_names'])

            # Get numeric columns, excluding ID columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not self._is_id_column(col)]

            if not numeric_cols:
                print(f"   No meaningful numeric columns in {table_name} (IDs excluded)")
                continue

            table_plots = {}

            # Create distribution plots for non-ID columns
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                if len(df[col].dropna()) > 0:
                    # Distribution plot
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    # Histogram
                    axes[0].hist(df[col].dropna(), bins=20, edgecolor='black', alpha=0.7)
                    axes[0].set_title(f'Distribution of {col}')
                    axes[0].set_xlabel(col)
                    axes[0].set_ylabel('Frequency')
                    axes[0].grid(True, alpha=0.3)

                    # Box plot
                    axes[1].boxplot(df[col].dropna())
                    axes[1].set_title(f'Box Plot of {col}')
                    axes[1].set_ylabel(col)
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()

                    # Convert to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    plot_base64 = base64.b64encode(buffer.read()).decode()
                    plt.close()

                    table_plots[f'distribution_{col}'] = plot_base64

            # Create scatter plots for pairs of non-ID numeric columns
            if len(numeric_cols) >= 2:
                # Take first few pairs
                pairs = [(numeric_cols[i], numeric_cols[j])
                        for i in range(min(3, len(numeric_cols)))
                        for j in range(i+1, min(4, len(numeric_cols)))][:3]

                for col1, col2 in pairs:
                    fig, ax = plt.subplots(figsize=(8, 6))

                    # Remove any NaN values
                    valid_data = df[[col1, col2]].dropna()

                    if len(valid_data) > 0:
                        ax.scatter(valid_data[col1], valid_data[col2], alpha=0.6)
                        ax.set_xlabel(col1)
                        ax.set_ylabel(col2)
                        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                        ax.grid(True, alpha=0.3)

                        # Add trend line if enough points
                        if len(valid_data) > 3:
                            z = np.polyfit(valid_data[col1], valid_data[col2], 1)
                            p = np.poly1d(z)
                            ax.plot(valid_data[col1], p(valid_data[col1]),
                                  "r--", alpha=0.5, label='Trend line')
                            ax.legend()

                        plt.tight_layout()

                        buffer = BytesIO()
                        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                        buffer.seek(0)
                        plot_base64 = base64.b64encode(buffer.read()).decode()
                        plt.close()

                        table_plots[f'scatter_{col1}_vs_{col2}'] = plot_base64

            # Create combined box plot for all non-ID numeric columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))

                # Prepare data for box plots
                data_to_plot = []
                labels = []

                for col in numeric_cols[:8]:  # Limit to 8 columns for readability
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Normalize data for better comparison (z-score normalization)
                        if col_data.std() > 0:
                            normalized = (col_data - col_data.mean()) / col_data.std()
                            data_to_plot.append(normalized)
                            labels.append(col[:15])  # Truncate long names

                if data_to_plot:
                    ax.boxplot(data_to_plot, labels=labels)
                    ax.set_title(f'Normalized Box Plots Comparison - {table_name}')
                    ax.set_ylabel('Z-Score (normalized values)')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Mean')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    plot_base64 = base64.b64encode(buffer.read()).decode()
                    plt.close()

                    table_plots['combined_boxplot'] = plot_base64

            plots[table_name] = table_plots

        return plots

    def _display_stats_in_notebook(self, stats_summary, plot_artifacts):
        """Display statistics and plots directly in notebook"""
        import base64

        print("\n📈 Statistical Analysis Results")
        print("=" * 60)

        for table_name, stats in stats_summary.items():
            print(f"\n📊 Table: {table_name}")
            print(f"Total Rows: {stats['total_rows']}")

            # Display numeric column statistics
            if stats['numeric_columns']:
                print("\nNumeric Columns Statistics:")

                # Create HTML table for stats
                html = """
                <table style='border-collapse: collapse; margin: 10px 0;'>
                <tr style='background-color: #f0f0f0;'>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Column</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Min</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Max</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Mean</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Median</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Std Dev</th>
                </tr>
                """

                for col, col_stats in stats['numeric_columns'].items():
                    html += f"""
                    <tr>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col}</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col_stats['min']:.2f}</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col_stats['max']:.2f}</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col_stats['mean']:.2f}</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col_stats['median']:.2f}</td>
                        <td style='padding: 8px; border: 1px solid #ddd;'>{col_stats['std']:.2f}</td>
                    </tr>
                    """

                html += "</table>"
                display(HTML(html))

            # Display plots for this table
            if table_name in plot_artifacts and plot_artifacts[table_name]:
                print(f"\n📊 Visualizations for {table_name}:")

                for plot_name, plot_base64 in plot_artifacts[table_name].items():
                    # Display the plot
                    img_html = f'<img src="data:image/png;base64,{plot_base64}" style="max-width:100%;">'
                    display(HTML(img_html))

            print("-" * 60)

    def _generate_stats_html(self, stats_summary, plot_artifacts):
        """Generate standalone HTML with statistics and plots"""
        html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <title>Statistical Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2E86AB; }
        h2 { color: #444; margin-top: 30px; }
        .stats-table { border-collapse: collapse; margin: 20px 0; }
        .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        .stats-table th { background-color: #f0f0f0; }
        .plot-container { margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>📊 Statistical Analysis Report</h1>
''']

        for table_name, stats in stats_summary.items():
            html_parts.append(f'''
    <h2>{table_name}</h2>
    <p>Total Rows: {stats['total_rows']}</p>
''')

            # Add numeric statistics table
            if stats['numeric_columns']:
                html_parts.append('''
    <h3>Numeric Columns Statistics</h3>
    <table class="stats-table">
        <tr>
            <th>Column</th><th>Min</th><th>Max</th><th>Mean</th>
            <th>Median</th><th>Std Dev</th><th>Q25</th><th>Q75</th>
        </tr>''')

                for col, col_stats in stats['numeric_columns'].items():
                    html_parts.append(f'''
        <tr>
            <td style="text-align:left;">{col}</td>
            <td>{col_stats['min']:.2f}</td>
            <td>{col_stats['max']:.2f}</td>
            <td>{col_stats['mean']:.2f}</td>
            <td>{col_stats['median']:.2f}</td>
            <td>{col_stats['std']:.2f}</td>
            <td>{col_stats['q25']:.2f}</td>
            <td>{col_stats['q75']:.2f}</td>
        </tr>''')

                html_parts.append('    </table>')

            # Add plots
            if table_name in plot_artifacts:
                html_parts.append(f'    <h3>Visualizations</h3>')
                for plot_name, plot_base64 in plot_artifacts[table_name].items():
                    plot_title = plot_name.replace('_', ' ').title()
                    html_parts.append(f'''
    <div class="plot-container">
        <h4>{plot_title}</h4>
        <img src="data:image/png;base64,{plot_base64}">
    </div>''')

        html_parts.append('''
</body>
</html>''')

        return ''.join(html_parts)

    # ... [Keep all the existing methods from the original VisualizationAgent] ...
    # _find_database, _get_tables_info, _generate_interactive_table_view,
    # _generate_data_explorer, _create_jupyter_widgets, _in_notebook,
    # _save_visualization_artifacts remain the same

    def _find_database(self, state):
        """Find the database file from state or filesystem"""
        from pathlib import Path

        # Check implementation artifacts for database info
        impl_artifacts = state.get("implementation_artifacts", {})
        db_info = impl_artifacts.get("database_info", {})

        if db_info.get("database_file"):
            db_path = Path(db_info["database_file"])
            if db_path.exists():
                print(f"   Found database: {db_path}")
                return db_path

        # Check project directory
        project_id = state.get("project_id")
        if project_id:
            possible_paths = [
                Path(f"{project_id}_db.db"),
                Path(f"{project_id}.db"),
            ]

            if self.artifact_manager:
                project_root = self.artifact_manager.project_path
                possible_paths.extend([
                    project_root / f"{project_id}_db.db",
                    project_root / f"{project_id}.db"
                ])

            for path in possible_paths:
                if path.exists():
                    print(f"   Found database: {path}")
                    return path

        print("   ❌ Could not find database")
        return None

    def _get_tables_info(self):
        """Get information about all tables in the database"""
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        tables_info = {}

        for (table_name,) in tables:
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
            rows = cursor.fetchall()

            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]

            tables_info[table_name] = {
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "sample_rows": rows,
                "column_names": [col[1] for col in columns],
                "total_count": total_count
            }

        return tables_info

    def _generate_interactive_table_view(self, tables_info):
        """Generate HTML with DataTables for interactive table viewing"""

        html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <title>Database Table Viewer</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2E86AB; }
        h2 { color: #444; margin-top: 30px; }
        .table-container { margin: 20px 0; }
        .dataTables_wrapper { margin: 20px 0; }
        .tab-buttons { margin: 20px 0; }
        .tab-button {
            padding: 10px 20px;
            margin-right: 10px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            cursor: pointer;
            border-radius: 5px;
        }
        .tab-button.active { background: #2E86AB; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>📊 Database Table Viewer</h1>
    <div class="tab-buttons">''']

        # Add tab buttons
        for i, table_name in enumerate(tables_info.keys()):
            active = "active" if i == 0 else ""
            html_parts.append(f'''
        <button class="tab-button {active}" onclick="showTable('{table_name}')">{table_name}</button>''')

        html_parts.append('''
    </div>''')

        # Add content for each table
        for i, (table_name, info) in enumerate(tables_info.items()):
            active = "active" if i == 0 else ""

            html_parts.append(f'''
    <div id="{table_name}" class="tab-content {active}">
        <h2>{table_name}</h2>
        <div class="stats">
            Total Records: {info['total_count']} | Columns: {len(info['columns'])}
        </div>
        <table id="table_{table_name}" class="display" style="width:100%">
            <thead>
                <tr>''')

            # Add column headers
            for col in info['column_names']:
                html_parts.append(f'''
                    <th>{col}</th>''')

            html_parts.append('''
                </tr>
            </thead>
            <tbody>''')

            # Add rows
            for row in info['sample_rows']:
                html_parts.append('''
                <tr>''')
                for value in row:
                    html_parts.append(f'''
                    <td>{value}</td>''')
                html_parts.append('''
                </tr>''')

            html_parts.append(f'''
            </tbody>
        </table>
    </div>''')

        # Add JavaScript
        html_parts.append('''
    <script>
        $(document).ready(function() {''')

        # Initialize DataTables for each table
        for table_name in tables_info.keys():
            html_parts.append(f'''
            $('#table_{table_name}').DataTable({{
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                searching: true,
                ordering: true,
                paging: true,
                info: true
            }});''')

        html_parts.append('''
        });

        function showTable(tableName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tableName).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>''')

        return ''.join(html_parts)

    def _generate_data_explorer(self, tables_info):
        """Generate a more advanced data explorer with filtering capabilities"""

        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Advanced Data Explorer</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/searchbuilder/1.3.1/css/searchBuilder.dataTables.min.css">

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.print.min.js"></script>
    <script src="https://cdn.datatables.net/searchbuilder/1.3.1/js/dataTables.searchBuilder.min.js"></script>

    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; }
        h1 { color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }
        .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .filter-box { background: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .filter-title { font-weight: bold; color: #495057; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Advanced Data Explorer</h1>
        <p>Use the Search Builder to create complex filters. Export data as CSV or print.</p>
        '''

        # Add explorer interface for first table (can be extended)
        if tables_info:
            first_table = list(tables_info.keys())[0]
            info = tables_info[first_table]

            html += f'''
        <h2>{first_table}</h2>
        <table id="explorer_table" class="display" style="width:100%">
            <thead>
                <tr>'''

            for col in info['column_names']:
                html += f'<th>{col}</th>'

            html += '''</tr>
            </thead>
            <tbody>'''

            for row in info['sample_rows']:
                html += '<tr>'
                for value in row:
                    html += f'<td>{value}</td>'
                html += '</tr>'

            html += '''</tbody>
        </table>
    </div>

    <script>
        $(document).ready(function() {
            $('#explorer_table').DataTable({
                dom: 'QBfrtip',
                buttons: [
                    'copy', 'csv', 'excel', 'pdf', 'print'
                ],
                searchBuilder: {
                    conditions: {
                        number: {
                            'between': {
                                conditionName: 'Between'
                            }
                        }
                    }
                },
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]]
            });
        });
    </script>
</body>
</html>'''

        return html

    def _create_jupyter_widgets(self, tables_info):
        """Create interactive Jupyter widgets for table viewing"""
        try:
            import pandas as pd

            # Create tab widget
            tab_contents = []
            tab_titles = []

            for table_name, info in tables_info.items():
                # Create DataFrame
                df = pd.DataFrame(info['sample_rows'], columns=info['column_names'])

                # Create widgets for this table
                search_box = widgets.Text(
                    placeholder=f'Search in {table_name}...',
                    description='Search:',
                    layout=widgets.Layout(width='300px')
                )

                column_dropdown = widgets.Dropdown(
                    options=['All'] + info['column_names'],
                    value='All',
                    description='Column:',
                    layout=widgets.Layout(width='200px')
                )

                sort_dropdown = widgets.Dropdown(
                    options=info['column_names'],
                    value=info['column_names'][0],
                    description='Sort by:',
                    layout=widgets.Layout(width='200px')
                )

                output = widgets.Output()

                def create_filter_function(df, output, search_box, column_dropdown, sort_dropdown):
                    def filter_data(*args):
                        with output:
                            output.clear_output()
                            filtered_df = df.copy()

                            # Apply search filter
                            search_term = search_box.value
                            if search_term:
                                if column_dropdown.value == 'All':
                                    mask = pd.Series([False] * len(filtered_df))
                                    for col in filtered_df.columns:
                                        mask |= filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                                    filtered_df = filtered_df[mask]
                                else:
                                    filtered_df = filtered_df[
                                        filtered_df[column_dropdown.value].astype(str).str.contains(search_term, case=False, na=False)
                                    ]

                            # Apply sorting
                            filtered_df = filtered_df.sort_values(by=sort_dropdown.value)

                            # Display
                            display(filtered_df)
                            print(f"Showing {len(filtered_df)} of {len(df)} records")

                    return filter_data

                # Create filter function for this table
                filter_func = create_filter_function(df, output, search_box, column_dropdown, sort_dropdown)

                # Set up event handlers
                search_box.observe(filter_func, names='value')
                column_dropdown.observe(filter_func, names='value')
                sort_dropdown.observe(filter_func, names='value')

                # Initial display
                with output:
                    display(df)
                    print(f"Showing {len(df)} records (sample of {info['total_count']} total)")

                # Create layout for this table
                controls = widgets.HBox([search_box, column_dropdown, sort_dropdown])
                table_widget = widgets.VBox([
                    widgets.HTML(f"<h3>Total Records: {info['total_count']}</h3>"),
                    controls,
                    output
                ])

                tab_contents.append(table_widget)
                tab_titles.append(table_name)

            # Create tabs
            tabs = widgets.Tab(children=tab_contents)
            for i, title in enumerate(tab_titles):
                tabs.set_title(i, title)

            return tabs

        except ImportError:
            print("   ⚠️ ipywidgets not available - skipping Jupyter widgets")
            return None

    def _in_notebook(self):
        """Check if running in Jupyter notebook"""
        try:
            if 'IPKernelApp' in get_ipython().config:
                return True
        except:
            pass
        return False

    def _save_visualization_artifacts(self, artifacts):
        """Save visualization files"""
        viz_dir = self.artifact_manager.project_path / "08_visualizations"
        viz_dir.mkdir(exist_ok=True)

        saved_files = []

        # Save table view HTML
        if artifacts.get("table_view_html"):
            file_path = viz_dir / "table_viewer.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(artifacts["table_view_html"])
            saved_files.append(str(file_path))
            print(f"   📁 Saved interactive table viewer: {file_path}")

        # Save data explorer HTML
        if artifacts.get("data_explorer_html"):
            file_path = viz_dir / "data_explorer.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(artifacts["data_explorer_html"])
            saved_files.append(str(file_path))
            print(f"   📁 Saved advanced data explorer: {file_path}")

        self.artifact_manager.created_files.extend(saved_files)
        return saved_files

# ============================================================================
# CONVERSATIONAL QUERY AGENT
# ============================================================================

import re
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

class ConversationalQueryAgent:
    """
    Enables natural language querying of databases created by Data Architect Agent.
    Converts natural language questions to SQL and returns formatted results.
    """

    def __init__(self, llm=None, artifact_manager=None, database_path=None):
        """
        Initialize the Conversational Query Agent

        Args:
            llm: Language model for NL to SQL conversion
            artifact_manager: Access to project artifacts
            database_path: Path to the database file
        """
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.database_path = database_path
        self.connection = None

        # Schema information
        self.schema_info = {}
        self.table_relationships = {}
        self.glossary = {}
        self.conversation_history = []

        # Load metadata if available
        if self.artifact_manager:
            self._load_metadata()

    def execute(self, state):
        """Execute the conversational query agent"""
        print("💬 Initializing Conversational Query Interface...")

        # Find and connect to database
        db_path = self._find_database(state)
        if not db_path:
            state["errors"].append("No database found for querying")
            return state

        self.database_path = db_path

        try:
            # Connect to database
            self.connection = sqlite3.connect(str(db_path))

            # Load schema information
            self._load_schema_info()

            # Load relationships from models if available
            self._load_relationships(state)

            # Create conversational interface
            if self._in_notebook():
                query_interface = self._create_notebook_interface()
                state["query_interface"] = query_interface

                # Display the interface
                print("\n💬 Natural Language Query Interface Ready!")
                display(query_interface)
            else:
                # Create CLI interface
                cli_interface = self._create_cli_interface()
                state["query_interface"] = cli_interface

            state["conversational_query_ready"] = True
            state["current_step"] = "conversational_query_ready"

            return state

        except Exception as e:
            print(f"   ❌ Error initializing query interface: {e}")
            state["errors"].append(f"Query interface failed: {str(e)}")
            return state

    def _load_metadata(self):
        """Load project metadata for enhanced query understanding"""
        if not self.artifact_manager:
            return

        # Load glossary for domain terminology
        glossary_file = self.artifact_manager.project_path / "07_documentation" / "glossary.json"
        if glossary_file.exists():
            with open(glossary_file, 'r') as f:
                glossary_data = json.load(f)
                for entry in glossary_data.get('entries', []):
                    self.glossary[entry['term'].lower()] = entry['definition']

        # Load conceptual model for entity understanding
        conceptual_file = self.artifact_manager.project_path / "02_models" / "conceptual_model.xmi"
        if conceptual_file.exists():
            # Parse for entity relationships
            pass  # Simplified for now

    def _find_database(self, state):
        """Find the database file"""
        from pathlib import Path

        # Check implementation artifacts
        impl_artifacts = state.get("implementation_artifacts", {})
        db_info = impl_artifacts.get("database_info", {})

        if db_info.get("database_file"):
            db_path = Path(db_info["database_file"])
            if db_path.exists():
                return db_path

        # Check project directory
        project_id = state.get("project_id")
        if project_id:
            possible_paths = [
                Path(f"{project_id}_db.db"),
                Path(f"{project_id}.db"),
            ]

            for path in possible_paths:
                if path.exists():
                    return path

        return None

    def _load_schema_info(self):
        """Load complete schema information from database"""
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()

            # Get sample data for context
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()

            # Store schema info
            self.schema_info[table_name] = {
                'columns': [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'default': col[4],
                        'primary_key': bool(col[5])
                    }
                    for col in columns
                ],
                'foreign_keys': [
                    {
                        'column': fk[3],
                        'references_table': fk[2],
                        'references_column': fk[4]
                    }
                    for fk in foreign_keys
                ],
                'sample_data': sample_data[:3] if sample_data else []
            }

    def _load_relationships(self, state):
        """Load relationship information from models"""
        # Extract from logical or conceptual models if available
        logical_model = state.get("logical_model")
        if logical_model:
            # Parse relationships from XMI
            # Simplified for now - would parse actual XMI
            pass

    def process_natural_language_query(self, nl_query: str) -> Dict[str, Any]:
        """
        Convert natural language query to SQL and execute

        Args:
            nl_query: Natural language question

        Returns:
            Dictionary with SQL, results, and explanation
        """
        try:
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'query': nl_query,
                'type': 'user'
            })

            # Generate SQL from natural language
            sql_query = self._generate_sql(nl_query)

            # Validate SQL
            validation = self._validate_sql(sql_query)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'suggestion': self._suggest_correction(nl_query)
                }

            # Execute query
            results = self._execute_query(sql_query)

            # Format results
            formatted_results = self._format_results(results, sql_query)

            # Generate explanation
            explanation = self._generate_explanation(nl_query, sql_query, results)

            # Add to history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'sql': sql_query,
                'results': formatted_results,
                'type': 'response'
            })

            return {
                'success': True,
                'natural_language': nl_query,
                'sql': sql_query,
                'results': formatted_results,
                'explanation': explanation,
                'row_count': len(results) if results else 0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'natural_language': nl_query
            }

    def _generate_sql(self, nl_query: str) -> str:
        """Generate SQL from natural language using LLM"""

        # Build schema context
        schema_context = self._build_schema_context()

        if self.llm:
            prompt = f"""Convert this natural language query to SQL:

Natural Language: {nl_query}

Database Schema:
{schema_context}

Instructions:
1. Generate a valid SQL query for SQLite
2. Use proper JOIN syntax when multiple tables are needed
3. Include appropriate WHERE clauses
4. Use column aliases for clarity
5. Limit results to 100 rows unless specified otherwise

Return ONLY the SQL query, no explanations.
"""

            try:
                response = self.llm.invoke(prompt)
                sql = response.content if hasattr(response, 'content') else str(response)

                # Clean up SQL
                sql = sql.strip()
                if sql.startswith('```sql'):
                    sql = sql[6:]
                if sql.startswith('```'):
                    sql = sql[3:]
                if sql.endswith('```'):
                    sql = sql[:-3]

                return sql.strip()

            except Exception as e:
                print(f"LLM SQL generation failed: {e}")
                return self._generate_fallback_sql(nl_query)
        else:
            return self._generate_fallback_sql(nl_query)

    def _build_schema_context(self) -> str:
        """Build a concise schema description for the LLM"""
        context_parts = []

        for table_name, info in self.schema_info.items():
            columns = [f"{col['name']} ({col['type']})" for col in info['columns']]
            context_parts.append(f"Table: {table_name}")
            context_parts.append(f"  Columns: {', '.join(columns)}")

            if info['foreign_keys']:
                for fk in info['foreign_keys']:
                    context_parts.append(
                        f"  FK: {fk['column']} -> {fk['references_table']}.{fk['references_column']}"
                    )

        return '\n'.join(context_parts)

    def _generate_fallback_sql(self, nl_query: str) -> str:
        """Generate basic SQL without LLM"""
        nl_lower = nl_query.lower()

        # Find mentioned tables
        mentioned_tables = []
        for table in self.schema_info.keys():
            if table.lower() in nl_lower or table[:-1].lower() in nl_lower:  # Handle singular/plural
                mentioned_tables.append(table)

        if not mentioned_tables:
            # Default to first table
            mentioned_tables = [list(self.schema_info.keys())[0]]

        table = mentioned_tables[0]

        # Determine query type
        if any(word in nl_lower for word in ['count', 'how many', 'number of']):
            return f"SELECT COUNT(*) as count FROM {table}"
        elif any(word in nl_lower for word in ['average', 'avg', 'mean']):
            # Find numeric column
            numeric_cols = [col['name'] for col in self.schema_info[table]['columns']
                          if 'int' in col['type'].lower() or 'real' in col['type'].lower()]
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]}) as average FROM {table}"
        elif any(word in nl_lower for word in ['maximum', 'max', 'highest']):
            numeric_cols = [col['name'] for col in self.schema_info[table]['columns']
                          if 'int' in col['type'].lower() or 'real' in col['type'].lower()]
            if numeric_cols:
                return f"SELECT MAX({numeric_cols[0]}) as maximum FROM {table}"
        elif any(word in nl_lower for word in ['show', 'list', 'get', 'find']):
            return f"SELECT * FROM {table} LIMIT 20"
        else:
            return f"SELECT * FROM {table} LIMIT 10"

    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL query before execution"""
        sql_upper = sql.upper()

        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return {
                    'valid': False,
                    'error': f"Query contains restricted operation: {keyword}"
                }

        # Check if tables exist
        mentioned_tables = self._extract_tables_from_sql(sql)
        for table in mentioned_tables:
            if table.lower() not in [t.lower() for t in self.schema_info.keys()]:
                return {
                    'valid': False,
                    'error': f"Table '{table}' not found in database"
                }

        return {'valid': True}

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []

        # Simple regex patterns for table extraction
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.extend(matches)

        return list(set(tables))

    def _execute_query(self, sql: str) -> List[Tuple]:
        """Execute SQL query and return results"""
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def _format_results(self, results: List[Tuple], sql: str) -> Any:
        """Format query results for display"""
        if not results:
            return "No results found"

        # Get column names from query
        cursor = self.connection.cursor()
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]

        # Create DataFrame for better formatting
        df = pd.DataFrame(results, columns=columns)

        # Limit display size
        if len(df) > 20:
            return {
                'preview': df.head(20).to_dict('records'),
                'total_rows': len(df),
                'columns': columns,
                'truncated': True
            }
        else:
            return {
                'data': df.to_dict('records'),
                'total_rows': len(df),
                'columns': columns,
                'truncated': False
            }

    def _generate_explanation(self, nl_query: str, sql: str, results: List) -> str:
        """Generate human-readable explanation of the query and results"""

        if self.llm:
            prompt = f"""Explain this database query and its results in simple terms:

Question: {nl_query}
SQL Query: {sql}
Number of Results: {len(results) if results else 0}

Provide a brief, clear explanation of:
1. What the query does
2. What the results show
3. Any important insights

Keep it concise (2-3 sentences).
"""

            try:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except:
                pass

        # Fallback explanation
        return f"Query executed successfully. Found {len(results) if results else 0} results."

    def _suggest_correction(self, nl_query: str) -> str:
        """Suggest query corrections or alternatives"""
        suggestions = []

        # Check if any table names are close to query terms
        nl_lower = nl_query.lower()
        for table in self.schema_info.keys():
            if any(part in table.lower() for part in nl_lower.split()):
                suggestions.append(f"Try: 'Show all {table}'")

        if not suggestions:
            suggestions = [
                f"Try: 'Show all {list(self.schema_info.keys())[0]}'",
                "Try: 'Count total records'",
                "Try: 'Show recent entries'"
            ]

        return " | ".join(suggestions[:3])

    def _create_notebook_interface(self):
        """Create Jupyter notebook interface for queries"""
        try:

            # Create widgets
            query_input = widgets.Textarea(
                placeholder='Enter your question in plain English...\nExamples:\n- Show all customers\n- How many orders were placed last month?\n- What is the average order value?',
                layout=widgets.Layout(width='100%', height='80px')
            )

            submit_button = widgets.Button(
                description='Ask Question',
                button_style='primary',
                icon='search'
            )

            clear_button = widgets.Button(
                description='Clear',
                button_style='warning',
                icon='trash'
            )

            output_area = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    height='400px',
                    border='1px solid #ddd',
                    overflow='auto'
                )
            )

            # History display
            history_output = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    height='200px',
                    border='1px solid #eee',
                    overflow='auto'
                )
            )

            # Sample queries dropdown
            sample_queries = widgets.Dropdown(
                options=[
                    'Select a sample query...',
                    'Show all tables in the database',
                    'Count total records in each table',
                    'Show recent entries',
                    'Find top 10 by amount',
                    'Show unique values in status column',
                    'Calculate average and sum of numeric columns'
                ],
                value='Select a sample query...',
                layout=widgets.Layout(width='50%')
            )

            def on_sample_selected(change):
                if change['new'] != 'Select a sample query...':
                    query_input.value = change['new']
                    sample_queries.value = 'Select a sample query...'

            sample_queries.observe(on_sample_selected, names='value')

            # Submit handler
            def on_submit(b=None):
                query = query_input.value.strip()
                if not query:
                    return

                with output_area:
                    output_area.clear_output()
                    print(f"🔍 Query: {query}")
                    print("-" * 50)

                    # Process query
                    result = self.process_natural_language_query(query)

                    if result['success']:
                        # Show SQL
                        print(f"📝 Generated SQL:")
                        print(f"   {result['sql']}")
                        print()

                        # Show explanation
                        if result.get('explanation'):
                            print(f"📊 Explanation:")
                            print(f"   {result['explanation']}")
                            print()

                        # Show results
                        print(f"📋 Results ({result['row_count']} rows):")

                        if isinstance(result['results'], dict):
                            if result['results'].get('truncated'):
                                df = pd.DataFrame(result['results']['preview'])
                                display(df)
                                print(f"\n... showing first 20 of {result['results']['total_rows']} rows")
                            else:
                                df = pd.DataFrame(result['results']['data'])
                                display(df)
                        else:
                            print(result['results'])
                    else:
                        print(f"❌ Error: {result['error']}")
                        if result.get('suggestion'):
                            print(f"💡 Suggestion: {result['suggestion']}")

                # Update history
                with history_output:
                    history_output.clear_output()
                    print("📜 Query History:")
                    for item in self.conversation_history[-5:]:
                        if item['type'] == 'user':
                            print(f"  Q: {item['query']}")

            def on_clear(b):
                query_input.value = ''
                output_area.clear_output()
                with output_area:
                    print("Ready for new query...")

            # Connect handlers
            submit_button.on_click(on_submit)
            clear_button.on_click(on_clear)
            # REMOVED: query_input.on_submit(on_submit) - Textarea doesn't support this

            # Create layout
            controls = widgets.HBox([submit_button, clear_button, sample_queries])

            # Schema info accordion
            schema_items = []
            for table_name, info in self.schema_info.items():
                columns_text = '\n'.join([
                    f"  • {col['name']} ({col['type']})"
                    for col in info['columns']
                ])
                schema_widget = widgets.HTML(f"<pre>Columns:\n{columns_text}</pre>")
                schema_items.append(schema_widget)

            schema_accordion = widgets.Accordion(children=schema_items)
            for i, table_name in enumerate(self.schema_info.keys()):
                schema_accordion.set_title(i, f"📊 {table_name}")
            schema_accordion.selected_index = None

            # Main interface
            interface = widgets.VBox([
                widgets.HTML('<h3>🔍 Natural Language Database Query Interface</h3>'),
                widgets.HTML('<p>Ask questions about your data in plain English!</p>'),
                query_input,
                controls,
                output_area,
                widgets.HTML('<h4>Database Schema</h4>'),
                schema_accordion,
                widgets.HTML('<h4>Recent Queries</h4>'),
                history_output
            ])

            # Show initial message
            with output_area:
                print("💬 Query Interface Ready!")
                print(f"📊 Connected to database with {len(self.schema_info)} tables")
                print("\nTry asking questions like:")
                print("  • 'Show all customers'")
                print("  • 'How many orders do we have?'")
                print("  • 'What's the average price?'")

            return interface

        except ImportError:
            print("   ⚠️ ipywidgets not available - CLI mode only")
            return None

    def _create_cli_interface(self):
        """Create command-line interface for queries"""

        def cli_loop():
            print("\n" + "="*60)
            print("💬 Natural Language Query Interface")
            print("="*60)
            print(f"Connected to database with {len(self.schema_info)} tables")
            print("\nType 'help' for commands, 'exit' to quit")
            print("-"*60)

            while True:
                try:
                    query = input("\n🔍 Your question: ").strip()

                    if query.lower() == 'exit':
                        print("Goodbye!")
                        break
                    elif query.lower() == 'help':
                        self._show_cli_help()
                    elif query.lower() == 'schema':
                        self._show_schema()
                    elif query.lower() == 'history':
                        self._show_history()
                    else:
                        result = self.process_natural_language_query(query)
                        self._display_cli_result(result)

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")

        return cli_loop

    def _show_cli_help(self):
        """Show CLI help"""
        print("\n📚 Available Commands:")
        print("  help    - Show this help")
        print("  schema  - Show database schema")
        print("  history - Show query history")
        print("  exit    - Exit the interface")
        print("\n💡 Example Questions:")
        print("  • Show all customers")
        print("  • Count orders by status")
        print("  • What's the total revenue?")

    def _show_schema(self):
        """Display database schema"""
        print("\n📊 Database Schema:")
        for table_name, info in self.schema_info.items():
            print(f"\n  Table: {table_name}")
            for col in info['columns']:
                pk = " [PK]" if col['primary_key'] else ""
                print(f"    • {col['name']} ({col['type']}){pk}")

    def _show_history(self):
        """Display query history"""
        print("\n📜 Recent Queries:")
        for item in self.conversation_history[-10:]:
            if item['type'] == 'user':
                print(f"  Q: {item['query']}")

    def _display_cli_result(self, result):
        """Display query result in CLI"""
        if result['success']:
            print(f"\n✅ SQL: {result['sql']}")
            print(f"📊 Results: {result['row_count']} rows")

            if isinstance(result['results'], dict):
                df = pd.DataFrame(
                    result['results'].get('preview') or result['results'].get('data')
                )
                print(df.to_string())

                if result['results'].get('truncated'):
                    print(f"\n... showing first 20 of {result['results']['total_rows']} rows")
            else:
                print(result['results'])

            if result.get('explanation'):
                print(f"\n💡 {result['explanation']}")
        else:
            print(f"\n❌ Error: {result['error']}")
            if result.get('suggestion'):
                print(f"💡 Try: {result['suggestion']}")

    def _in_notebook(self):
        """Check if running in Jupyter notebook"""
        try:
            if 'IPKernelApp' in get_ipython().config:
                return True
        except:
            pass
        return False

# ============================================================================
# REST API IMPLEMENTATION AGENT
# ============================================================================

import json
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import textwrap

class RESTAPIImplementationAgent:
    """Generates fully functional REST API implementations from OpenAPI specifications"""

    def __init__(self, llm=None, artifact_manager=None, db_config=None):
        """
        Initialize the REST API Implementation Agent

        Args:
            llm: Language model (optional, for enhanced generation)
            artifact_manager: Artifact manager for saving files
            db_config: Database configuration
        """
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.db_config = db_config or self._get_default_db_config()
        self.api_spec = None
        self.database_name = None
        self.tables = []

    def _get_default_db_config(self):
        """Get default database configuration"""
        return {
            'type': 'sqlite',
            'database_file': None  # Will be set based on project
        }

    def execute(self, state):
        """Execute the REST API implementation generation"""
        print("🌐 REST API Implementation Agent Starting...")

        # Get project information
        project_id = state.get("project_id", "data_architect_project")
        self.database_name = f"{project_id}_db"

        # Get API specification from state or files
        api_spec = self._get_api_specification(state)
        if not api_spec:
            state["errors"].append("No API specification found")
            return state

        self.api_spec = api_spec

        # Get database information
        db_info = self._get_database_info(state)

        # Extract table schemas from DDL
        sql_ddl = state.get("sql_ddl") or state.get("implementation_artifacts", {}).get("sql_ddl")
        if sql_ddl:
            self.tables = self._parse_tables_from_ddl(sql_ddl)

        try:
            # Generate REST API implementation
            print("   1️⃣ Generating FastAPI application...")
            api_implementation = self._generate_fastapi_implementation(api_spec, db_info)

            print("   2️⃣ Generating database models...")
            db_models = self._generate_database_models()

            print("   3️⃣ Generating utility functions...")
            utils = self._generate_utils()

            print("   4️⃣ Generating requirements file...")
            requirements = self._generate_requirements()

            print("   5️⃣ Generating Docker configuration...")
            docker_config = self._generate_docker_config()

            print("   6️⃣ Generating startup script...")
            startup_script = self._generate_startup_script()

            print("   7️⃣ Generating API client examples...")
            client_examples = self._generate_client_examples(api_spec)

            print("   8️⃣ Generating API documentation...")
            api_docs = self._generate_api_documentation(api_spec)

            # Package all artifacts
            api_artifacts = {
                "main.py": api_implementation,
                "models.py": db_models,
                "database.py": utils['database'],
                "schemas.py": utils['schemas'],
                "requirements.txt": requirements,
                "Dockerfile": docker_config['dockerfile'],
                "docker-compose.yml": docker_config['compose'],
                "start_api.sh": startup_script,
                "client_examples.py": client_examples,
                "API_DOCUMENTATION.md": api_docs,
                ".env.example": self._generate_env_example()
            }

            # Save artifacts if artifact manager available
            if self.artifact_manager:
                saved_files = self._save_api_artifacts(api_artifacts)
                state["api_implementation_files"] = saved_files

            # Update state
            state["api_implementation"] = api_artifacts
            state["api_implementation_status"] = "completed"
            state["current_step"] = "api_implementation_complete"

            print("   ✅ REST API implementation completed successfully!")
            print(f"   📁 Generated {len(api_artifacts)} files")
            print(f"   🚀 Run 'python main.py' to start the API server")

            return state

        except Exception as e:
            print(f"   ❌ Error generating API implementation: {e}")
            state["errors"].append(f"API implementation failed: {str(e)}")
            return state

    def _get_api_specification(self, state):
        """Get API specification from state or files"""
        # Try to get from state
        if state.get("data_product_config", {}).get("api_specification"):
            return state["data_product_config"]["api_specification"]

        # Try to read from file if artifact manager available
        if self.artifact_manager:
            api_file = self.artifact_manager.project_path / "05_data_products" / "api_specification.json"
            if api_file.exists():
                with open(api_file, 'r') as f:
                    return json.load(f)

        return None

    def _get_database_info(self, state):
        """Get database information from state"""
        db_info = state.get("implementation_artifacts", {}).get("database_info", {})

        if not db_info:
            # Default SQLite configuration
            db_info = {
                "database_type": "SQLite",
                "database_file": f"{self.database_name}.db"
            }

        return db_info

    def _parse_tables_from_ddl(self, ddl: str) -> List[Dict]:
        """Parse table structures from DDL"""
        tables = []

        # Simple regex parsing for CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        matches = re.findall(create_table_pattern, ddl, re.DOTALL | re.IGNORECASE)

        for table_name, columns_text in matches:
            columns = []

            # Parse columns
            column_lines = columns_text.split(',')
            for line in column_lines:
                line = line.strip()
                if line and not any(keyword in line.upper() for keyword in ['CONSTRAINT', 'PRIMARY KEY', 'FOREIGN KEY', 'CHECK']):
                    parts = line.split()
                    if len(parts) >= 2:
                        col_name = parts[0]
                        col_type = parts[1]
                        is_primary = 'PRIMARY KEY' in line.upper()
                        is_nullable = 'NOT NULL' not in line.upper()

                        columns.append({
                            'name': col_name,
                            'type': col_type,
                            'is_primary': is_primary,
                            'nullable': is_nullable
                        })

            tables.append({
                'name': table_name,
                'columns': columns
            })

        return tables

    def _generate_fastapi_implementation(self, api_spec: Dict, db_info: Dict) -> str:
        """Generate the main FastAPI application"""

        app_name = api_spec.get("info", {}).get("title", "Data API")
        app_version = api_spec.get("info", {}).get("version", "1.0.0")
        app_description = api_spec.get("info", {}).get("description", "RESTful API for data management")

        # Generate route implementations for each path
        routes = []
        for path, methods in api_spec.get("paths", {}).items():
            for method, details in methods.items():
                route_code = self._generate_route(path, method, details)
                routes.append(route_code)

        implementation = f'''"""
{app_name} - Auto-generated REST API Implementation
Generated by Data Architect Agent
Version: {app_version}
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv

# Import our modules
from database import get_db, init_db
from models import *
from schemas import *

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="{app_name}",
    description="""{app_description}""",
    version="{app_version}",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup"""
    init_db()
    print("✅ Database initialized")
    print(f"🚀 API server running at http://localhost:8000")
    print(f"📚 Interactive docs at http://localhost:8000/docs")

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Check API health status"""
    return {{
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "{app_version}"
    }}

# ============================================================================
# API ENDPOINTS
# ============================================================================

{"".join(routes)}

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={{"error": "Resource not found", "path": str(request.url)}}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={{"error": "Internal server error"}}
    )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    print(f"Starting {app_name} on {{host}}:{{port}}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )
'''
        return implementation

    def _generate_route(self, path: str, method: str, details: Dict) -> str:
        """Generate a single route implementation"""

        summary = details.get("summary", "API endpoint")
        description = details.get("description", "")

        # Extract resource name from path
        resource = self._extract_resource_from_path(path)

        # Generate function name
        func_name = self._generate_function_name(path, method)

        # Generate appropriate implementation based on method
        if method.lower() == "get":
            if "{id}" in path or "{" in path:
                return self._generate_get_by_id_route(path, resource, summary, description)
            else:
                return self._generate_list_route(path, resource, summary, description)
        elif method.lower() == "post":
            return self._generate_create_route(path, resource, summary, description)
        elif method.lower() == "put":
            return self._generate_update_route(path, resource, summary, description)
        elif method.lower() == "delete":
            return self._generate_delete_route(path, resource, summary, description)
        else:
            return f"\n# TODO: Implement {method.upper()} {path}\n"

    def _extract_resource_from_path(self, path: str) -> str:
        """Extract resource name from API path"""
        # Remove leading slash and parameters
        clean_path = path.lstrip('/').split('/')[0].split('{')[0]
        # Convert to singular if plural
        if clean_path.endswith('s'):
            return clean_path[:-1]
        return clean_path

    def _generate_function_name(self, path: str, method: str) -> str:
        """Generate function name from path and method"""
        resource = self._extract_resource_from_path(path)

        if "{id}" in path or "{" in path:
            if method.lower() == "get":
                return f"get_{resource}_by_id"
            elif method.lower() == "put":
                return f"update_{resource}"
            elif method.lower() == "delete":
                return f"delete_{resource}"
        else:
            if method.lower() == "get":
                return f"list_{resource}s"
            elif method.lower() == "post":
                return f"create_{resource}"

        return f"{method.lower()}_{resource}"

    def _generate_list_route(self, path: str, resource: str, summary: str, description: str) -> str:
        """Generate a GET list endpoint"""

        resource_plural = resource + "s"
        resource_class = resource.title()

        return f'''
@app.get("{path}", response_model=List[{resource_class}Schema], tags=["{resource_class}"])
async def list_{resource_plural}(
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(100, description="Number of items to return"),
    search: Optional[str] = Query(None, description="Search term"),
    db = Depends(get_db)
):
    """
    {summary}

    {description}
    """
    try:
        query = db.query({resource_class})

        # Apply search filter if provided
        if search:
            query = query.filter(
                {resource_class}.name.contains(search) |
                {resource_class}.description.contains(search)
            )

        # Apply pagination
        {resource_plural} = query.offset(skip).limit(limit).all()

        return {resource_plural}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving {resource_plural}: {{str(e)}}"
        )
'''

    def _generate_get_by_id_route(self, path: str, resource: str, summary: str, description: str) -> str:
        """Generate a GET by ID endpoint"""

        resource_class = resource.title()

        return f'''
@app.get("{path}", response_model={resource_class}Schema, tags=["{resource_class}"])
async def get_{resource}_by_id(
    id: int = Path(..., description="{resource_class} ID"),
    db = Depends(get_db)
):
    """
    {summary}

    {description}
    """
    {resource} = db.query({resource_class}).filter({resource_class}.id == id).first()

    if not {resource}:
        raise HTTPException(
            status_code=404,
            detail=f"{resource_class} with ID {{id}} not found"
        )

    return {resource}
'''

    def _generate_create_route(self, path: str, resource: str, summary: str, description: str) -> str:
        """Generate a POST create endpoint"""

        resource_class = resource.title()

        return f'''
@app.post("{path}", response_model={resource_class}Schema, status_code=status.HTTP_201_CREATED, tags=["{resource_class}"])
async def create_{resource}(
    {resource}_data: {resource_class}CreateSchema = Body(...),
    db = Depends(get_db)
):
    """
    {summary}

    {description}
    """
    try:
        # Create new {resource}
        new_{resource} = {resource_class}(**{resource}_data.dict())

        # Add to database
        db.add(new_{resource})
        db.commit()
        db.refresh(new_{resource})

        return new_{resource}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Error creating {resource}: {{str(e)}}"
        )
'''

    def _generate_update_route(self, path: str, resource: str, summary: str, description: str) -> str:
        """Generate a PUT update endpoint"""

        resource_class = resource.title()

        return f'''
@app.put("{path}", response_model={resource_class}Schema, tags=["{resource_class}"])
async def update_{resource}(
    id: int = Path(..., description="{resource_class} ID"),
    {resource}_data: {resource_class}UpdateSchema = Body(...),
    db = Depends(get_db)
):
    """
    {summary}

    {description}
    """
    # Find existing {resource}
    {resource} = db.query({resource_class}).filter({resource_class}.id == id).first()

    if not {resource}:
        raise HTTPException(
            status_code=404,
            detail=f"{resource_class} with ID {{id}} not found"
        )

    try:
        # Update fields
        update_data = {resource}_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr({resource}, field, value)

        # Update timestamp
        {resource}.updated_at = datetime.now()

        db.commit()
        db.refresh({resource})

        return {resource}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Error updating {resource}: {{str(e)}}"
        )
'''

    def _generate_delete_route(self, path: str, resource: str, summary: str, description: str) -> str:
        """Generate a DELETE endpoint"""

        resource_class = resource.title()

        return f'''
@app.delete("{path}", status_code=status.HTTP_204_NO_CONTENT, tags=["{resource_class}"])
async def delete_{resource}(
    id: int = Path(..., description="{resource_class} ID"),
    db = Depends(get_db)
):
    """
    {summary}

    {description}
    """
    # Find existing {resource}
    {resource} = db.query({resource_class}).filter({resource_class}.id == id).first()

    if not {resource}:
        raise HTTPException(
            status_code=404,
            detail=f"{resource_class} with ID {{id}} not found"
        )

    try:
        db.delete({resource})
        db.commit()

        return None  # 204 No Content

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Error deleting {resource}: {{str(e)}}"
        )
'''

    def _generate_database_models(self) -> str:
        """Generate SQLAlchemy database models"""

        models = ['"""', 'Database Models', 'Auto-generated by Data Architect Agent', '"""', '',
                  'from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Numeric, Date',
                  'from sqlalchemy.ext.declarative import declarative_base',
                  'from sqlalchemy.orm import relationship',
                  'from datetime import datetime', '',
                  'Base = declarative_base()', '']

        # Generate model for each table
        for table in self.tables:
            model_code = self._generate_model_class(table)
            models.append(model_code)
            models.append('')

        return '\n'.join(models)

    def _generate_model_class(self, table: Dict) -> str:
        """Generate a SQLAlchemy model class for a table"""

        table_name = table['name']
        class_name = self._table_to_class_name(table_name)

        model_lines = [
            f"class {class_name}(Base):",
            f'    """Model for {table_name} table"""',
            f'    __tablename__ = "{table_name}"',
            ''
        ]

        # Add columns
        for column in table['columns']:
            col_def = self._generate_column_definition(column)
            model_lines.append(f'    {col_def}')

        # Add repr method
        model_lines.extend([
            '',
            '    def __repr__(self):',
            f'        return f"<{class_name}(id={{self.id}}, name={{self.name}})>"'
        ])

        return '\n'.join(model_lines)

    def _table_to_class_name(self, table_name: str) -> str:
        """Convert table name to class name"""
        # Remove plural 's' and capitalize
        if table_name.endswith('ies'):
            singular = table_name[:-3] + 'y'
        elif table_name.endswith('s'):
            singular = table_name[:-1]
        else:
            singular = table_name

        # Convert to PascalCase
        return ''.join(word.capitalize() for word in singular.split('_'))

    def _generate_column_definition(self, column: Dict) -> str:
        """Generate SQLAlchemy column definition"""

        col_name = column['name']
        col_type = column['type']

        # Map SQL types to SQLAlchemy types
        type_map = {
            'SERIAL': 'Integer',
            'INTEGER': 'Integer',
            'VARCHAR': 'String(255)',
            'TEXT': 'Text',
            'TIMESTAMP': 'DateTime',
            'DATE': 'Date',
            'DECIMAL': 'Numeric(10, 2)',
            'BOOLEAN': 'Boolean'
        }

        # Get SQLAlchemy type
        sa_type = 'String(255)'  # Default
        for sql_type, sa_type_name in type_map.items():
            if sql_type in col_type.upper():
                sa_type = sa_type_name
                break

        # Build column definition
        col_def = f"{col_name} = Column({sa_type}"

        if column.get('is_primary'):
            col_def += ", primary_key=True"
            if 'SERIAL' in col_type.upper():
                col_def += ", autoincrement=True"

        if not column.get('nullable', True):
            col_def += ", nullable=False"

        if col_name == 'created_at':
            col_def += ", default=datetime.now"
        elif col_name == 'updated_at':
            col_def += ", default=datetime.now, onupdate=datetime.now"

        col_def += ")"

        return col_def

    def _generate_utils(self) -> Dict[str, str]:
        """Generate utility modules"""

        # Database connection module
        database_module = '''"""
Database Connection and Session Management
Auto-generated by Data Architect Agent
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os
from models import Base

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./''' + self.database_name + '''.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

        # Pydantic schemas module
        schemas_module = self._generate_schemas()

        return {
            'database': database_module,
            'schemas': schemas_module
        }

    def _generate_schemas(self) -> str:
        """Generate Pydantic schemas for validation"""

        schemas = ['"""', 'Pydantic Schemas for Request/Response Validation',
                   'Auto-generated by Data Architect Agent', '"""', '',
                   'from pydantic import BaseModel, Field, validator',
                   'from typing import Optional, List',
                   'from datetime import datetime', '']

        # Generate schemas for each table
        for table in self.tables:
            schema_code = self._generate_schema_classes(table)
            schemas.append(schema_code)
            schemas.append('')

        return '\n'.join(schemas)

    def _generate_schema_classes(self, table: Dict) -> str:
        """Generate Pydantic schema classes for a table"""

        table_name = table['name']
        class_name = self._table_to_class_name(table_name)

        # Base schema with common fields
        base_schema = [
            f"class {class_name}Base(BaseModel):",
            '    """Base schema for ' + class_name + '"""'
        ]

        # Add fields (excluding auto-generated ones)
        for column in table['columns']:
            if column['name'] not in ['id', 'created_at', 'updated_at']:
                field_def = self._generate_field_definition(column)
                base_schema.append(f'    {field_def}')

        # Create schema for creation
        create_schema = [
            '',
            f"class {class_name}CreateSchema({class_name}Base):",
            f'    """Schema for creating {class_name}"""',
            '    pass'
        ]

        # Update schema
        update_schema = [
            '',
            f"class {class_name}UpdateSchema(BaseModel):",
            f'    """Schema for updating {class_name}"""'
        ]

        # Make all fields optional for update
        for column in table['columns']:
            if column['name'] not in ['id', 'created_at', 'updated_at']:
                field_def = self._generate_field_definition(column, optional=True)
                update_schema.append(f'    {field_def}')

        # Response schema
        response_schema = [
            '',
            f"class {class_name}Schema({class_name}Base):",
            f'    """Schema for {class_name} response"""',
            '    id: int',
            '    created_at: datetime',
            '    updated_at: datetime',
            '',
            '    class Config:',
            '        orm_mode = True'
        ]

        return '\n'.join(base_schema + create_schema + update_schema + response_schema)

    def _generate_field_definition(self, column: Dict, optional: bool = False) -> str:
        """Generate Pydantic field definition"""

        col_name = column['name']
        col_type = column['type']

        # Map SQL types to Python types
        type_map = {
            'INTEGER': 'int',
            'VARCHAR': 'str',
            'TEXT': 'str',
            'DECIMAL': 'float',
            'BOOLEAN': 'bool',
            'DATE': 'datetime',
            'TIMESTAMP': 'datetime'
        }

        # Get Python type
        py_type = 'str'  # Default
        for sql_type, python_type in type_map.items():
            if sql_type in col_type.upper():
                py_type = python_type
                break

        # Build field definition
        if optional or column.get('nullable', True):
            field_def = f"{col_name}: Optional[{py_type}] = None"
        else:
            field_def = f"{col_name}: {py_type}"

        return field_def

    def _generate_requirements(self) -> str:
        """Generate requirements.txt file"""

        requirements = [
            "# REST API Requirements",
            "# Auto-generated by Data Architect Agent",
            "",
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "sqlalchemy==2.0.23",
            "pydantic==2.5.0",
            "python-dotenv==1.0.0",
            "python-multipart==0.0.6",
            "",
            "# Database drivers",
            "# Uncomment the one you need:",
            "# psycopg2-binary==2.9.9  # PostgreSQL",
            "# pymysql==1.1.0  # MySQL",
            "# For SQLite, no additional driver needed",
            "",
            "# Optional: For development",
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
            "httpx==0.25.2",  # For testing
            "",
            "# Optional: For production",
            "gunicorn==21.2.0",
            "prometheus-client==0.19.0",  # For metrics
            "python-json-logger==2.0.7"  # For structured logging
        ]

        return '\n'.join(requirements)

    def _generate_docker_config(self) -> Dict[str, str]:
        """Generate Docker configuration files"""

        dockerfile = '''# Dockerfile for REST API
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

        docker_compose = f'''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./data/{self.database_name}.db
      - API_HOST=0.0.0.0
      - API_PORT=8000
    volumes:
      - ./data:/app/data
    restart: unless-stopped
'''

        return {
            'dockerfile': dockerfile,
            'compose': docker_compose
        }

    def _generate_startup_script(self) -> str:
        """Generate startup script for the API"""

        script = '''#!/bin/bash
# Startup script for REST API
# Auto-generated by Data Architect Agent

echo "🚀 Starting REST API Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables (if .env exists)
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# Run the application
echo "🌐 Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
'''

        return script

    def _generate_client_examples(self, api_spec: Dict) -> str:
        """Generate client code examples"""

        base_url = "http://localhost:8000"

        examples = [
            '"""',
            'API Client Examples',
            'Auto-generated by Data Architect Agent',
            '"""',
            '',
            'import requests',
            'import json',
            'from typing import Dict, List, Optional',
            '',
            f'BASE_URL = "{base_url}"',
            '',
            'class APIClient:',
            '    """Simple API client for testing"""',
            '    ',
            '    def __init__(self, base_url: str = BASE_URL):',
            '        self.base_url = base_url',
            '        self.session = requests.Session()',
            '    '
        ]

        # Generate methods for each endpoint
        for path, methods in api_spec.get("paths", {}).items():
            for method, details in methods.items():
                client_method = self._generate_client_method(path, method, details)
                examples.append(client_method)
                examples.append('')

        # Add example usage
        examples.extend([
            '',
            'if __name__ == "__main__":',
            '    # Example usage',
            '    client = APIClient()',
            '    ',
            '    # Health check',
            '    print("🏥 Health Check:")',
            '    health = client.session.get(f"{BASE_URL}/health").json()',
            '    print(json.dumps(health, indent=2))',
            '    ',
            '    # Example: List items',
            '    # response = client.list_items(limit=10)',
            '    # print(json.dumps(response, indent=2))',
            '    ',
            '    # Example: Create item',
            '    # new_item = {"name": "Test Item", "description": "Test description"}',
            '    # response = client.create_item(new_item)',
            '    # print(json.dumps(response, indent=2))'
        ])

        return '\n'.join(examples)

    def _generate_client_method(self, path: str, method: str, details: Dict) -> str:
        """Generate a client method for an endpoint"""

        resource = self._extract_resource_from_path(path)
        func_name = self._generate_function_name(path, method)

        if method.lower() == "get":
            if "{id}" in path:
                return f'''    def {func_name}(self, id: int) -> Dict:
        """Get {resource} by ID"""
        response = self.session.get(f"{{self.base_url}}{path.replace("{id}", "{id}")}")
        response.raise_for_status()
        return response.json()'''
            else:
                return f'''    def {func_name}(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """List {resource}s"""
        params = {{"skip": skip, "limit": limit}}
        response = self.session.get(f"{{self.base_url}}{path}", params=params)
        response.raise_for_status()
        return response.json()'''

        elif method.lower() == "post":
            return f'''    def {func_name}(self, data: Dict) -> Dict:
        """Create new {resource}"""
        response = self.session.post(f"{{self.base_url}}{path}", json=data)
        response.raise_for_status()
        return response.json()'''

        elif method.lower() == "put":
            return f'''    def {func_name}(self, id: int, data: Dict) -> Dict:
        """Update {resource}"""
        response = self.session.put(f"{{self.base_url}}{path.replace("{id}", "{id}")}", json=data)
        response.raise_for_status()
        return response.json()'''

        elif method.lower() == "delete":
            return f'''    def {func_name}(self, id: int) -> None:
        """Delete {resource}"""
        response = self.session.delete(f"{{self.base_url}}{path.replace("{id}", "{id}")}")
        response.raise_for_status()'''

        return f"    # TODO: Implement {method} {path}"

    def _generate_api_documentation(self, api_spec: Dict) -> str:
        """Generate comprehensive API documentation"""

        app_name = api_spec.get("info", {}).get("title", "Data API")
        app_version = api_spec.get("info", {}).get("version", "1.0.0")
        app_description = api_spec.get("info", {}).get("description", "RESTful API")

        docs = [
            f"# {app_name} Documentation",
            "",
            f"**Version:** {app_version}",
            f"**Description:** {app_description}",
            "",
            "## Quick Start",
            "",
            "### Installation",
            "",
            "```bash",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "",
            "# Run the server",
            "python main.py",
            "```",
            "",
            "### Docker",
            "",
            "```bash",
            "# Build and run with Docker Compose",
            "docker-compose up --build",
            "```",
            "",
            "## API Endpoints",
            "",
            "Base URL: `http://localhost:8000`",
            "",
            "### Interactive Documentation",
            "",
            "- Swagger UI: http://localhost:8000/docs",
            "- ReDoc: http://localhost:8000/redoc",
            "",
            "### Available Endpoints",
            ""
        ]

        # Document each endpoint
        for path, methods in api_spec.get("paths", {}).items():
            docs.append(f"#### `{path}`")
            docs.append("")

            for method, details in methods.items():
                summary = details.get("summary", "")
                description = details.get("description", "")

                docs.append(f"**{method.upper()}** - {summary}")
                if description:
                    docs.append(f"  {description}")
                docs.append("")

        # Add more sections
        docs.extend([
            "## Authentication",
            "",
            "Currently, the API does not require authentication. For production use, implement:",
            "- JWT tokens",
            "- API keys",
            "- OAuth 2.0",
            "",
            "## Error Handling",
            "",
            "The API uses standard HTTP status codes:",
            "",
            "- `200` - Success",
            "- `201` - Created",
            "- `204` - No Content",
            "- `400` - Bad Request",
            "- `404` - Not Found",
            "- `500` - Internal Server Error",
            "",
            "## Database",
            "",
            f"The API uses SQLite database: `{self.database_name}.db`",
            "",
            "To use a different database, update the `DATABASE_URL` environment variable.",
            "",
            "## Development",
            "",
            "### Testing",
            "",
            "```python",
            "# Run client examples",
            "python client_examples.py",
            "```",
            "",
            "### Environment Variables",
            "",
            "Create a `.env` file with:",
            "",
            "```",
            "DATABASE_URL=sqlite:///./your_database.db",
            "API_HOST=0.0.0.0",
            "API_PORT=8000",
            "```"
        ])

        return '\n'.join(docs)

    def _generate_env_example(self) -> str:
        """Generate .env.example file"""

        return f"""# Environment Variables
# Copy to .env and update values

# Database
DATABASE_URL=sqlite:///./{self.database_name}.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost/dbname
# For MySQL: DATABASE_URL=mysql+pymysql://user:password@localhost/dbname

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Security (add for production)
# SECRET_KEY=your-secret-key-here
# API_KEY=your-api-key-here

# CORS (for production, specify allowed origins)
# CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
"""

    def _save_api_artifacts(self, artifacts: Dict[str, str]) -> List[str]:
        """Save API implementation artifacts to files"""

        if not self.artifact_manager:
            return []

        # Create API implementation directory
        api_dir = self.artifact_manager.project_path / "03_implementation" / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for filename, content in artifacts.items():
            file_path = api_dir / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Make shell scripts executable
            if filename.endswith('.sh'):
                os.chmod(file_path, 0o755)

            saved_files.append(str(file_path))
            self.artifact_manager.created_files.append(str(file_path))

        print(f"   📁 API implementation saved to: {api_dir}")

        return saved_files

import pandas as pd
import sqlite3
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from io import StringIO

class DDLWithCSVReverseEngineeringAgent:
    """
    Enhanced reverse engineering agent that processes DDL and populates
    the database with CSV data
    """

    def __init__(self, llm, artifact_manager=None):
        self.llm = llm
        self.artifact_manager = artifact_manager
        self.parser = SimpleDDLParser()
        self.connection = None
        self.database_path = None

    def execute(self, state):
        """Reverse engineer models from DDL and populate with CSV data"""
        print("🔄 DDL + CSV Reverse Engineering Agent Starting...")

        ddl_content = state.get("ddl_content")
        csv_files = state.get("csv_files", {})

        if not ddl_content:
            state["errors"].append("No DDL content provided")
            return state

        if not csv_files:
            state["errors"].append("No CSV files provided")
            return state

        try:
            # Step 1: Parse DDL using existing parser
            tables = self.parser.parse_ddl(ddl_content)

            if not tables:
                state["errors"].append("No tables found in DDL")
                return state

            print(f"   Found {len(tables)} tables: {', '.join(t.name for t in tables)}")

            # Step 2: Create database and load DDL
            self._create_and_populate_database(ddl_content, csv_files, state.get("project_id"))

            # Step 3: Generate models (same as original reverse engineering)
            conceptual_model = self._generate_conceptual_model_from_tables(tables)
            conceptual_dbml = self._generate_dbml_from_tables(tables, "Conceptual")

            logical_model = self._generate_logical_model_from_tables(tables)
            logical_dbml = self._generate_dbml_from_tables(tables, "Logical")

            physical_model = self._generate_physical_model_from_ddl(tables, ddl_content)

            # Step 4: Extract requirements with CSV context
            requirements = self._extract_requirements_from_tables_and_csv(tables, csv_files)

            # CRITICAL: Set entities at state level for downstream agents
            if requirements.get("entities"):
                state["inferred_entities"] = requirements["entities"]
                print(f"   ✅ Extracted entities: {', '.join(requirements['entities'])}")
                print(f"   ✅ Set inferred_entities in state for downstream agents")

            # Step 5: Save all artifacts
            if self.artifact_manager:
                self.artifact_manager.save_model(conceptual_model, "conceptual")
                self.artifact_manager.save_dbml(conceptual_dbml, "conceptual")
                self.artifact_manager.save_model(logical_model, "logical")
                self.artifact_manager.save_dbml(logical_dbml, "logical")
                self.artifact_manager.save_model(physical_model, "physical")
                self.artifact_manager.save_requirements(requirements)

                # Save original DDL and CSV info
                impl_artifacts = {
                    "sql_ddl": ddl_content,
                    "database_info": {
                        "database_type": "SQLite",
                        "database_file": str(self.database_path),
                        "csv_files_loaded": len(csv_files),
                        "tables_populated": len([t for t in tables if t.name.lower() in [f.lower().replace('.csv', '') for f in csv_files.keys()]])
                    }
                }
                self.artifact_manager.save_implementation_artifacts(impl_artifacts)

            # In DDLWithCSVReverseEngineeringAgent.execute, near the end:

            # Update state - ensure proper flags are set
            state["physical_model"] = physical_model
            state["logical_model"] = logical_model
            state["conceptual_model"] = conceptual_model
            state["conceptual_model_dbml"] = conceptual_dbml
            state["logical_model_dbml"] = logical_dbml
            state["requirements"] = requirements
            state["implementation_artifacts"] = impl_artifacts
            state["database_path"] = str(self.database_path)  # ADD THIS
            state["reverse_engineered"] = True
            state["data_loaded"] = True  # Critical flag
            state["reverse_engineered_with_data"] = True  # Additional flag for clarity
            state["current_step"] = "ddl_csv_reverse_engineering_complete"

            # Double-check entities are in state
            if requirements.get("entities"):
                state["inferred_entities"] = requirements["entities"]

            print("   ✅ DDL + CSV reverse engineering completed successfully!")
            print(f"   📊 Database created: {self.database_path}")
            print(f"   📝 CSV files processed: {len(csv_files)}")

            return state

        except Exception as e:
            print(f"   ❌ Error during DDL + CSV reverse engineering: {e}")
            import traceback
            traceback.print_exc()
            state["errors"].append(str(e))
            return state


    def _create_and_populate_database(self, ddl_content: str, csv_files: Dict[str, str], project_id: str):
        """Create database from DDL and populate with CSV data"""

        # Create database path
        if self.artifact_manager:
            project_root = self.artifact_manager.project_path.parent
            self.database_path = project_root / f"{project_id}_db.db"
        else:
            self.database_path = Path(f"{project_id}_db.db")

        # Remove existing database if it exists
        if self.database_path.exists():
            self.database_path.unlink()

        print(f"   🗄️ Creating database: {self.database_path}")

        # Create SQLite database
        self.connection = sqlite3.connect(str(self.database_path))
        cursor = self.connection.cursor()

        # Convert DDL to SQLite format and execute
        sqlite_ddl = self._convert_ddl_to_sqlite(ddl_content)
        statements = self._split_sql_statements(sqlite_ddl)

        tables_created = 0
        for statement in statements:
            if statement.strip():
                try:
                    cursor.execute(statement)
                    if 'CREATE TABLE' in statement.upper():
                        tables_created += 1
                except Exception as e:
                    print(f"   ⚠️ DDL statement warning: {e}")

        self.connection.commit()
        print(f"   ✅ Created {tables_created} tables")

        # Load CSV data
        self._load_csv_data_into_tables(csv_files)

    def _load_csv_data_into_tables(self, csv_files: Dict[str, str]):
        """Load CSV data into corresponding database tables"""

        cursor = self.connection.cursor()

        # Get list of tables in database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        db_tables = [row[0] for row in cursor.fetchall()]

        loaded_tables = []
        total_rows = 0

        for csv_filename, csv_content in csv_files.items():
            # Extract table name from CSV filename
            table_name = os.path.splitext(csv_filename)[0].lower()

            # Find matching table (case-insensitive)
            matching_table = None
            for db_table in db_tables:
                if db_table.lower() == table_name:
                    matching_table = db_table
                    break

            if not matching_table:
                print(f"   ⚠️ No matching table found for CSV file: {csv_filename}")
                continue

            try:
                # Load CSV content into pandas DataFrame
                df = pd.read_csv(StringIO(csv_content))

                # Clean column names (convert to lowercase, replace spaces with underscores)
                df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]

                # Get table schema
                cursor.execute(f"PRAGMA table_info({matching_table})")
                table_columns = [row[1].lower() for row in cursor.fetchall()]

                # Filter DataFrame to only include columns that exist in the table
                available_columns = [col for col in df.columns if col in table_columns]
                if not available_columns:
                    print(f"   ⚠️ No matching columns found for {csv_filename}")
                    continue

                df_filtered = df[available_columns]

                # Clean data for SQLite compatibility
                df_filtered = self._clean_dataframe_for_sqlite(df_filtered)

                # Insert data into table
                df_filtered.to_sql(matching_table, self.connection, if_exists='append', index=False)

                rows_loaded = len(df_filtered)
                total_rows += rows_loaded
                loaded_tables.append(matching_table)

                print(f"   ✅ Loaded {rows_loaded} rows into {matching_table}")

            except Exception as e:
                print(f"   ❌ Error loading {csv_filename}: {e}")

        self.connection.commit()
        print(f"   📊 Total: {total_rows} rows loaded into {len(loaded_tables)} tables")

    def _clean_dataframe_for_sqlite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for SQLite compatibility"""

        df_clean = df.copy()

        # Handle NaN values
        df_clean = df_clean.where(pd.notna(df_clean), None)

        # Convert datetime columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to datetime if it looks like a date
                sample_values = df_clean[col].dropna().head(5)
                if not sample_values.empty:
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    except:
                        pass

        # Convert boolean columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'bool':
                df_clean[col] = df_clean[col].astype(int)

        return df_clean

    def _convert_ddl_to_sqlite(self, ddl: str) -> str:
        """Convert DDL to SQLite compatible format"""
        sqlite_ddl = ddl

        # Remove unsupported features
        sqlite_ddl = sqlite_ddl.replace('SERIAL', 'INTEGER')
        sqlite_ddl = sqlite_ddl.replace('AUTO_INCREMENT', 'AUTOINCREMENT')
        sqlite_ddl = sqlite_ddl.replace('DEFAULT CURRENT_TIMESTAMP', "DEFAULT (datetime('now'))")
        sqlite_ddl = sqlite_ddl.replace('GENERATED ALWAYS AS', '--GENERATED ALWAYS AS')
        sqlite_ddl = sqlite_ddl.replace('CREATE EXTENSION', '-- CREATE EXTENSION')

        # Convert data types
        type_mappings = {
            'VARCHAR': 'TEXT',
            'DECIMAL': 'REAL',
            'TIMESTAMP': 'DATETIME',
            'BOOLEAN': 'INTEGER'
        }

        for pg_type, sqlite_type in type_mappings.items():
            sqlite_ddl = sqlite_ddl.replace(f'{pg_type}(', f'{sqlite_type}(')
            sqlite_ddl = sqlite_ddl.replace(f'{pg_type} ', f'{sqlite_type} ')

        return sqlite_ddl

    def _split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements"""
        lines = sql.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip().startswith('--'):
                cleaned_lines.append(line)

        cleaned_sql = '\n'.join(cleaned_lines)
        statements = cleaned_sql.split(';')
        return [stmt.strip() for stmt in statements if stmt.strip()]

    def _extract_requirements_from_tables_and_csv(self, tables: List, csv_files: Dict[str, str]) -> Dict[str, Any]:
        """Extract requirements considering both table structure and CSV data"""

        requirements = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "compliance_requirements": [],
            "entities": []  # Standardize on "entities" key
        }

        # Extract entity names properly
        entity_names = []
        for table in tables:
            entity = self._table_to_entity_name(table.name)
            entity_names.append(entity)
            requirements["functional_requirements"].append(f"Manage {entity} data")

        requirements["entities"] = entity_names

        # Analyze CSV data volume for scalability requirements
        total_records = 0
        for csv_content in csv_files.values():
            try:
                df = pd.read_csv(StringIO(csv_content))
                total_records += len(df)
            except:
                pass

        if total_records > 10000:
            requirements["non_functional_requirements"].append("High-volume data processing capability")
        if total_records > 1000:
            requirements["non_functional_requirements"].append("Efficient data indexing and querying")

        # Standard requirements
        requirements["non_functional_requirements"].extend([
            "Maintain referential integrity",
            "Support concurrent access",
            "Ensure data consistency"
        ])

        requirements["data_requirements"] = [
            f"Store and manage {len(tables)} entity types",
            f"Support bulk data loading from CSV files",
            f"Handle {total_records} initial records",
            "Support relationships between entities"
        ]

        requirements["compliance_requirements"].extend([
            "Data integrity validation",
            "Audit trail for data changes"
        ])

        return requirements

    def _table_to_entity_name(self, table_name: str) -> str:
        """Convert table name to entity name"""
        name = table_name.lower()
        if name.endswith('ies'):
            name = name[:-3] + 'y'
        elif name.endswith('s') and not name.endswith('ss'):
            name = name[:-1]

        return ''.join(word.capitalize() for word in name.split('_'))

    # Include existing methods from SimplifiedDDLReverseEngineeringAgent
    def _generate_conceptual_model_from_tables(self, tables):
        """Generate conceptual model XML from tables"""
        entities = []
        for table in tables:
            entity_name = self._table_to_entity_name(table.name)
            entities.append(entity_name)

        xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="ConceptualModel" type="Conceptual" source="ddl_csv_reverse_engineered">',
            '    <entities>'
        ]

        for table in tables:
            entity_name = self._table_to_entity_name(table.name)
            xmi_parts.extend([
                f'      <entity name="{entity_name}" description="Business entity for {table.name}">',
                f'        <attribute name="id" type="identifier"/>',
                f'        <attribute name="name" type="string"/>',
                f'        <attribute name="description" type="text"/>',
                f'        <attribute name="created_at" type="datetime"/>',
                f'        <attribute name="updated_at" type="datetime"/>'
            ])

            for col in table.columns:
                if col['name'].lower() not in ['id', 'name', 'description', 'created_at', 'updated_at']:
                    attr_type = self._map_sql_type_to_conceptual(col['type'])
                    xmi_parts.append(f'        <attribute name="{col["name"]}" type="{attr_type}"/>')

            xmi_parts.append(f'      </entity>')

        xmi_parts.extend([
            '    </entities>',
            '    <relationships>',
            '      <!-- Relationships inferred from foreign keys and data analysis -->',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(xmi_parts)

    def _generate_dbml_from_tables(self, tables, model_type: str = "Logical") -> str:
        """Generate DBML representation from tables"""
        dbml_parts = [
            f"// {model_type} Data Model in DBML format",
            f"// Generated from DDL + CSV reverse engineering",
            ""
        ]

        for table in tables:
            dbml_parts.append(f"Table {table.name} {{")

            for column in table.columns:
                col_type = self._map_sql_type_to_dbml(column['type'])

                constraints = []
                if column.get('is_primary_key'):
                    constraints.append('pk')
                if not column.get('nullable', True):
                    constraints.append('not null')

                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""

                dbml_parts.append(f"  {column['name']} {col_type}{constraint_str}")

            dbml_parts.extend(["}", ""])

        return '\n'.join(dbml_parts)

    def _generate_logical_model_from_tables(self, tables):
        """Generate logical model from tables"""
        logical_xmi_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">',
            '  <model name="LogicalModel" type="Logical" source="ddl_csv_reverse_engineered">',
            '    <tables>'
        ]

        for table in tables:
            logical_xmi_parts.append(f'      <table name="{table.name}">')

            for column in table.columns:
                nullable = 'true' if column.get('nullable', True) else 'false'
                pk = 'true' if column.get('is_primary_key') else 'false'

                logical_xmi_parts.append(
                    f'        <column name="{column["name"]}" '
                    f'type="{column["type"]}" nullable="{nullable}" '
                    f'primary_key="{pk}"/>'
                )

            logical_xmi_parts.append(f'      </table>')

        logical_xmi_parts.extend([
            '    </tables>',
            '    <relationships>',
            '      <!-- Foreign key relationships detected from DDL and data -->',
            '    </relationships>',
            '  </model>',
            '</xmi:XMI>'
        ])

        return '\n'.join(logical_xmi_parts)

    def _generate_physical_model_from_ddl(self, tables, ddl_content: str) -> str:
        """Generate physical model XMI from DDL"""
        table_count = len(tables)

        xmi_template = '''<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI">
  <model name="PhysicalModel" type="Physical" source="ddl_csv_reverse_engineered">
    <description>Physical model reverse engineered from SQL DDL and populated with CSV data ({table_count} tables)</description>
    <sql_ddl><![CDATA[
{sql_ddl}
    ]]></sql_ddl>
  </model>
</xmi:XMI>'''

        return xmi_template.format(
            table_count=table_count,
            sql_ddl=ddl_content
        )

    def _map_sql_type_to_conceptual(self, sql_type: str) -> str:
        """Map SQL types to conceptual types"""
        sql_type_lower = sql_type.lower()

        if any(t in sql_type_lower for t in ['varchar', 'text', 'char']):
            return 'string'
        elif any(t in sql_type_lower for t in ['int', 'serial']):
            return 'integer'
        elif any(t in sql_type_lower for t in ['decimal', 'numeric', 'float']):
            return 'number'
        elif 'bool' in sql_type_lower:
            return 'boolean'
        elif any(t in sql_type_lower for t in ['date', 'time']):
            return 'datetime'
        else:
            return 'string'

    def _map_sql_type_to_dbml(self, sql_type: str) -> str:
        """Map SQL types to DBML types"""
        sql_type_lower = sql_type.lower()

        if 'serial' in sql_type_lower:
            return 'integer'
        elif 'varchar' in sql_type_lower or 'text' in sql_type_lower:
            return 'varchar'
        elif 'int' in sql_type_lower:
            return 'integer'
        elif 'decimal' in sql_type_lower or 'numeric' in sql_type_lower:
            return 'decimal'
        elif 'timestamp' in sql_type_lower or 'datetime' in sql_type_lower:
            return 'timestamp'
        elif 'date' in sql_type_lower:
            return 'date'
        elif 'bool' in sql_type_lower:
            return 'boolean'
        else:
            return 'varchar'

class DataArchitectOrchestrator:
    """Main orchestrator that coordinates all agents"""

    class _DataInferenceStep:
        def __init__(self, artifact_manager):
            self.artifact_manager = artifact_manager

        def _discover_csvs(self, state):
            import glob
            from pathlib import Path
            csvs = (state.get("csv_files") or state.get("inference_csv_files") or [])
            if csvs:
                return list(csvs)

            candidates = []
            proj = getattr(self.artifact_manager, "project_path", None)
            if proj:
                candidates += glob.glob(str(Path(proj) / "**/*.csv"), recursive=True)
            candidates += glob.glob("/mnt/data/*.csv")
            candidates += glob.glob("/mnt/data/**/*.csv")
            seen, out = set(), []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out

        def execute(self, state):
            from pathlib import Path

            csv_files = self._discover_csvs(state)
            if not csv_files:
                print("ℹ️ Data Inference: no CSV files found – skipping step.")
                return state

            # Get project ID for database naming
            project_id = state.get("project_id", "data_project")

            out_dir = (self.artifact_manager.get_inference_dir()
                      if self.artifact_manager
                      else Path("./inference_artifacts"))

            # Pass project context to the agent
            agent = DataInferenceAgent(
                output_dir=str(out_dir),
                db_url=state.get("db_url"),
                prefer_mysql=False,
                echo_sql=False,
            )

            # Store project ID for database naming
            agent.project_id = project_id

            result = agent.run(csv_files)

            state["inference_result"] = result
            state["data_already_loaded"] = True
            
            # Fix #3: Add inferred_schema for workflow continuation
            state["inferred_schema"] = result.get("schema", {})
            
            # Fix #4: Capture database path from DataInferenceAgent
            if result.get("database", {}).get("database_path"):
                db_file = result["database"]["database_path"]
                state["database_path"] = str(db_file)
                print(f"   Orchestrator: Set database_path to: {db_file}")

            if result.get("entity_names"):
                state["inferred_entities"] = result["entity_names"]
                print(f"   Orchestrator: Setting inferred_entities in state: {result['entity_names']}")

                if not state.get("requirements"):
                    state["requirements"] = {}
                state["requirements"]["entities"] = result["entity_names"]
                print(f"   Orchestrator: Also added to requirements.entities")

            ddl_path = (result.get("artifacts", {}).get("ddl_mysql")
                        or result.get("artifacts", {}).get("ddl_sqlite"))
            if ddl_path and Path(ddl_path).exists():
                state["ddl_content"] = Path(ddl_path).read_text()

            if self.artifact_manager and hasattr(self.artifact_manager, "register_created_files"):
                paths = list(result.get("artifacts", {}).values())

                # Register the database file
                if hasattr(agent, 'project_id'):
                    project_root = Path(out_dir).parent
                    sqlite_db = project_root / f"{agent.project_id}_db.db"
                    if sqlite_db.exists():
                        paths.append(sqlite_db)

                self.artifact_manager.register_created_files(paths)

            return state

    def __init__(self, llm, project_config: Optional[ProjectConfig] = None):
        self.llm = llm
        self.base_llm = llm  # ADD THIS LINE
        self.project_config = project_config
        self.artifact_manager = SimpleArtifactManager(project_config) if project_config else None
        self.agents = self._build_agents()

    # def _build_agents(self) -> dict:
    #     """Construct the canonical agent registry exactly once."""
    #     return {
    #         "requirements": RequirementsAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "conceptual": ConceptualModelAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "logical": LogicalModelAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "physical": PhysicalModelAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "glossary": GlossaryAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "data_product": DataProductAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "ontology": OntologyAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "reverse": SimplifiedDDLReverseEngineeringAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "implementation": EnhancedImplementationAgent(self.llm, artifact_manager=self.artifact_manager),
    #         "data_inference": self._DataInferenceStep(self.artifact_manager),
    #     }

# Update the DataArchitectOrchestrator class

    def _build_agents(self) -> dict:
      """Construct the canonical agent registry exactly once."""
      temp_wrapper = TemperatureAdjustedLLM(self.base_llm, AGENT_TEMPERATURES)

      return {
          "requirements": RequirementsAgent(
              temp_wrapper.get_llm_for_agent("requirements"),
              artifact_manager=self.artifact_manager
          ),
          "conceptual": ConceptualModelAgent(
              temp_wrapper.get_llm_for_agent("conceptual"),
              artifact_manager=self.artifact_manager
          ),
          "logical": LogicalModelAgent(
              temp_wrapper.get_llm_for_agent("logical"),
              artifact_manager=self.artifact_manager
          ),
          "physical": PhysicalModelAgent(
              temp_wrapper.get_llm_for_agent("physical"),
              artifact_manager=self.artifact_manager
          ),
          "glossary": GlossaryAgent(
              temp_wrapper.get_llm_for_agent("glossary"),
              artifact_manager=self.artifact_manager
          ),
          "data_product": DataProductAgent(
              temp_wrapper.get_llm_for_agent("data_product"),
              artifact_manager=self.artifact_manager
          ),
          "ontology": OntologyAgent(
              temp_wrapper.get_llm_for_agent("ontology"),
              artifact_manager=self.artifact_manager
          ),
          "reverse": SimplifiedDDLReverseEngineeringAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          ),
          "implementation": EnhancedImplementationAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          ),
          "ddl_csv_reverse": DDLWithCSVReverseEngineeringAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          ),
          "data_inference": self._DataInferenceStep(self.artifact_manager),
          "visualization": VisualizationAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          ),
          "conversational_query": ConversationalQueryAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          ),
          # ADD THIS:
          "raw_data": SchemaAwareRawDataAgent(
              self.llm,
              artifact_manager=self.artifact_manager
          )
      }



    # def execute_workflow(
    #     self,
    #     project_id: str,
    #     messages=None,
    #     csv_files: list[str] | None = None,
    #     db_url: str | None = None,
    #     reverse: bool = False,
    #     ddl_content: str = None,
    #     ddl_csv_data: dict = None,
    #     raw_data_content: bytes = None,      # ADD
    #     mapping_instructions: str = None     # ADD
    # ):
    #     """
    #     Unified workflow runner used by demos and the chatbot.
    #     """
    #     # Initial state
    #     state = {
    #         "project_id": project_id,
    #         "messages": messages or [],
    #         "errors": [],
    #         "current_step": "starting",
    #     }

    #     # Add CSV files if provided
    #     if csv_files:
    #         state["csv_files"] = list(csv_files)

    #     if db_url:
    #         state["db_url"] = db_url

    #     # Add DDL content if provided
    #     if ddl_content:
    #         state["ddl_content"] = ddl_content

    #     # Add CSV data for DDL reverse engineering if provided
    #     if ddl_csv_data:
    #         state["csv_files"] = ddl_csv_data

    #     # ADD THIS: Raw data inputs
    #     if raw_data_content:
    #         state["raw_data_content"] = raw_data_content
    #     if mapping_instructions:
    #         state["mapping_instructions"] = mapping_instructions

    #     print("\n🚀 Executing Data Architecture Workflow...")

    #     # Assemble the plan
    #     def _has(agent_name):
    #         return hasattr(self, "agents") and isinstance(self.agents, dict) and agent_name in self.agents

    #     # ADD THIS WORKFLOW BRANCH FIRST (before other reverse engineering checks):
    #     if raw_data_content and ddl_content and mapping_instructions and _has("raw_data"):
    #         print("📊 Raw Data Transformation Workflow")

    #         planned = []

    #         # Step 1: Transform raw data to CSV using schema
    #         planned.append(("raw_data", "Raw Data → CSV Transformation"))

    #         # Step 2: Use DDL+CSV reverse engineering with generated CSVs
    #         if _has("ddl_csv_reverse"):
    #             planned.append(("ddl_csv_reverse", "DDL + Generated CSV Reverse Engineering"))

    #         # Step 3: Continue with additional artifacts
    #         for name, label in [
    #             ("glossary", "Business Glossary"),
    #             ("data_product", "Data Product Spec"),
    #             ("ontology", "Ontology Draft"),
    #             ("visualization", "Data Visualization"),
    #             ("conversational_query", "Natural Language Query Interface")
    #         ]:
    #             if _has(name):
    #                 planned.append((name, label))

    #     # NEW: Check if this is the DDL+CSV provisioning workflow
    #     elif ddl_content and ddl_csv_data and _has("ddl_provisioning"):
    #         # ... existing DDL provisioning workflow ...
    #         pass

    #     else:
    #         # EXISTING WORKFLOWS - unchanged
    #         planned = []

    #         # 1) Data inference first (only if we have CSV files but no DDL)
    #         if csv_files and not ddl_content and _has("data_inference"):
    #             planned.append(("data_inference", "Data Inferencing"))

    #         # 2) Reverse engineering - choose the right agent
    #         if ddl_content:
    #             if ddl_csv_data and _has("ddl_csv_reverse"):
    #                 # Existing DDL+CSV reverse engineering agent
    #                 planned.append(("ddl_csv_reverse", "DDL + CSV Reverse Engineering"))
    #             elif _has("reverse"):
    #                 # Regular DDL reverse engineering
    #                 planned.append(("reverse", "Reverse Engineering (DDL → Models)"))

    #         # 3) Forward chain (rest remains the same)
    #         for name, label in [
    #             ("requirements", "Requirements Elicitation"),
    #             ("conceptual", "Conceptual Modeling"),
    #             ("logical", "Logical Modeling"),
    #             ("physical", "Physical Modeling"),
    #             ("glossary", "Business Glossary"),
    #             ("data_product", "Data Product Spec"),
    #             ("ontology", "Ontology Draft"),
    #             ("implementation", "Implementation / Provisioning"),
    #             ("visualization", "Data Visualization"),
    #             ("conversational_query", "Natural Language Query Interface")
    #         ]:
    #             if _has(name):
    #                 planned.append((name, label))

    #     # Filter runnable steps
    #     def _will_run(item):
    #         name, _ = item
    #         if name == "reverse":
    #             return reverse or bool(state.get("ddl_content"))
    #         if name == "implementation" and state.get("skip_implementation"):
    #             return False
    #         return True

    #     runnable = [p for p in planned if _will_run(p)]
    #     total = len(runnable)

    #     if total == 0:
    #         print("ℹ️ No runnable steps were found.")
    #         return state

    #     # Execute
    #     for idx, (name, label) in enumerate(runnable, start=1):
    #         print(f"{idx}/{total} - {label}...")
    #         state["current_step"] = name

    #         try:
    #             agent = self.agents[name]
    #             result = agent.execute(state)

    #             # Robust state handling
    #             if isinstance(result, dict):
    #                 if "errors" not in result:
    #                     result["errors"] = state.get("errors", [])

    #                 for key in ["project_id", "current_step"]:
    #                     if key not in result and key in state:
    #                         result[key] = state[key]

    #                 state = result

    #                 # ADD THIS: After raw_data agent, transfer generated CSVs to state
    #                 if name == "raw_data" and result.get("csv_generation_complete"):
    #                     csv_files = result.get("csv_files", {})
    #                     state["csv_files"] = csv_files
    #                     state["ddl_csv_data"] = csv_files
    #                     print(f"   ✅ Generated {len(csv_files)} CSV files for next step")

    #             print("   ✅ Done.")

    #         except Exception as e:
    #             err = f"{type(e).__name__}: {e}"
    #             print(f"   ❌ Workflow error: {err}")

    #             if "errors" not in state:
    #                 state["errors"] = []
    #             state["errors"].append(err)

    #     # Wrap up
    #     print("\n✅ WORKFLOW COMPLETED!")
    #     print(f"   📋 Project ID: {project_id}")
    #     print(f"   📊 Status: created")
    #     print(f"   📈 Final Step: {state.get('current_step', 'n/a')}")

    #     errors = state.get("errors", [])
    #     if errors:
    #         print(f"   ⚠️ Errors: {errors}")
    #     else:
    #         print("   🌟 No errors encountered.")

    #     return state


    def execute_workflow(
        self,
        project_id: str,
        messages=None,
        csv_files: list[str] | None = None,
        db_url: str | None = None,
        reverse: bool = False,
        ddl_content: str = None,
        ddl_csv_data: dict = None,
        initial_state: dict | None = None
    ):
        """
        Enhanced workflow execution that handles DDL + CSV directly
        """

        print(f"\n{'='*60}")
        print(f"Starting Data Architecture Workflow")
        print(f"Project: {project_id}")

        # Special case: DDL + CSV provided - skip to physical model and implementation
        if ddl_content and csv_files:
            print("Mode: DDL + CSV Direct Implementation")
            print(f"{'='*60}\n")

            state = {
                "project_id": project_id,
                "messages": messages or [],
                "current_step": "ddl_csv_direct",
                "errors": []
            }

            # Step 1: Store DDL as Physical Model
            print("📋 Step 1: Storing provided DDL as Physical Model...")
            state["physical_model"] = ddl_content
            state["sql_ddl"] = ddl_content

            # Save DDL to artifacts if manager exists
            if self.artifact_manager:
                ddl_file = self.artifact_manager.project_path / "02_models" / "physical_model.sql"
                ddl_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ddl_file, 'w', encoding='utf-8') as f:
                    f.write(ddl_content)
                print(f"   ✅ Saved DDL to {ddl_file}")

            # Step 2: Store CSV file paths
            print(f"\n📊 Step 2: Registering {len(csv_files)} CSV files...")
            state["csv_files"] = csv_files
            for csv_file in csv_files:
                file_name = csv_file.split('/')[-1] if '/' in csv_file else csv_file
                print(f"   ✅ {file_name}")

            # Step 3: Run Implementation Agent to create and populate database
            print("\n🔨 Step 3: Implementation (Database Creation & Data Loading)...")
            state["current_step"] = "implementation"

            result = self.agents["implementation"].execute(state)
            if isinstance(result, dict):
                state.update(result)

            # ADD THIS DIAGNOSTIC BLOCK:
            print("\n=== DIAGNOSTIC INFO ===")
            print(f"database_provisioned: {state.get('database_provisioned')}")
            print(f"database_path (root): {state.get('database_path')}")
            if state.get('implementation_artifacts'):
                db_info = state['implementation_artifacts'].get('database_info', {})
                print(f"database_file (nested): {db_info.get('database_file')}")
            print("======================\n")

            # Check if database was created successfully
            if state.get("database_provisioned"):
                print(f"   ✅ Database created: {state.get('database_path')}")

                # ADD THIS - Step 3.5: Data Products
                if state.get("physical_model"):
                    print("\n📡 Step 3.5: Data Products...")
                    # Ensure sql_ddl is in implementation_artifacts for the Data Product Agent
                    if "implementation_artifacts" not in state:
                        state["implementation_artifacts"] = {}
                    if "sql_ddl" not in state["implementation_artifacts"]:
                        state["implementation_artifacts"]["sql_ddl"] = state.get("ddl_content") or state.get("physical_model")
                    try:
                        dp_result = self.agents["data_product"].execute(state)
                        if isinstance(dp_result, dict):
                            state.update(dp_result)
                        print("   ✅ Data products generated")
                    except Exception as e:
                        print(f"   ⚠️ Data product generation skipped: {e}")
                        state["errors"].append(f"Data product generation failed: {e}")

                # Optional: Run visualization and query agents if database exists
                if state.get("database_path"):
                    # Step 4: Data Visualization (optional)
                    print("\n📊 Step 4: Data Visualization...")
                    state["current_step"] = "visualization"
                    try:
                        viz_result = self.agents["visualization"].execute(state)
                        if isinstance(viz_result, dict):
                            state.update(viz_result)
                        print("   ✅ Visualizations created")
                    except Exception as e:
                        print(f"   ⚠️ Visualization skipped: {e}")
                        state["errors"].append(f"Visualization failed: {e}")

                    # Step 5: Query Interface (optional)
                    print("\n💬 Step 5: Natural Language Query Interface...")
                    state["current_step"] = "conversational_query"
                    try:
                        query_result = self.agents["conversational_query"].execute(state)
                        if isinstance(query_result, dict):
                            state.update(query_result)
                        print("   ✅ Query interface ready")
                    except Exception as e:
                        print(f"   ⚠️ Query interface skipped: {e}")
                        state["errors"].append(f"Query interface failed: {e}")
            else:
                print("   ❌ Database provisioning failed")
                state["errors"].append("Database provisioning failed")

            # Complete
            state["final_step"] = state["current_step"]
            print(f"\n{'='*60}")
            print("✅ DDL + CSV Workflow Complete!")
            if state.get("database_path"):
                print(f"📁 Database: {state['database_path']}")
            if state.get("errors"):
                print(f"⚠️ Warnings: {', '.join(state['errors'])}")
            print(f"{'='*60}\n")

            return state

        # Normal workflow paths (existing logic)
        print(f"Mode: {'Reverse Engineering' if reverse else 'Forward Engineering'}")
        print(f"{'='*60}\n")

        # Use initial_state if provided (for explicit entities from chatbot)
        if initial_state:
            state = initial_state
            # Add additional fields
            state.update({
                "csv_files": csv_files,
                "db_url": db_url,
                "reverse": reverse
            })
            if "inferred_entities" in state:
                print(f"   ✅ Using explicit entities from chatbot: {', '.join(state['inferred_entities'])}")
        else:
            state = {
                "project_id": project_id,
                "messages": messages or [],
                "current_step": "starting",
                "csv_files": csv_files,
                "db_url": db_url,
                "reverse": reverse,
                "errors": []
            }

        if reverse:
            # Reverse engineering workflow
            if db_url:
                # Reverse from database
                print("🔄 Starting reverse engineering from database...")
                state["current_step"] = "reverse_engineering"

                result = self.agents["reverse_engineering"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

                # Continue to conceptual model
                if state.get("physical_model"):
                    print("\n📐 Creating conceptual model...")
                    state["current_step"] = "conceptual"
                    result = self.agents["conceptual"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

            elif csv_files:
                # Reverse from CSV files
                print("🔄 Starting reverse engineering from CSV files...")
                state["current_step"] = "data_inference"

                result = self.agents["data_inference"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

                # Continue through the pipeline
                if state.get("inferred_schema"):
                    # Requirements
                    print("\n📋 Generating requirements...")
                    state["current_step"] = "requirements"
                    result = self.agents["requirements"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

                    # Conceptual model
                    print("\n📐 Creating conceptual model...")
                    state["current_step"] = "conceptual"
                    result = self.agents["conceptual"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

                    # Logical model
                    print("\n🗂️ Creating logical model...")
                    state["current_step"] = "logical"
                    result = self.agents["logical"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

                    # Physical model
                    print("\n💾 Creating physical model...")
                    state["current_step"] = "physical"
                    result = self.agents["physical"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

                    # Implementation
                    print("\n🔨 Implementing database...")
                    state["current_step"] = "implementation"
                    result = self.agents["implementation"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)

                    # Data products
                    print("\n📊 Generating data products...")
                    state["current_step"] = "data_product"
                    try:
                        result = self.agents["data_product"].execute(state)
                        if isinstance(result, dict):
                            state.update(result)
                    except Exception as e:
                        print(f"   ⚠️ Data products skipped: {e}")
                        state["errors"].append(f"Data products failed: {e}")

                    # Visualization
                    if state.get("database_path"):
                        print("\n📈 Creating visualizations...")
                        state["current_step"] = "visualization"
                        try:
                            result = self.agents["visualization"].execute(state)
                            if isinstance(result, dict):
                                state.update(result)
                        except Exception as e:
                            print(f"   ⚠️ Visualization skipped: {e}")
                            state["errors"].append(f"Visualization failed: {e}")

                    # Query interface
                    if state.get("database_path"):
                        print("\n💬 Setting up query interface...")
                        state["current_step"] = "conversational_query"
                        try:
                            result = self.agents["conversational_query"].execute(state)
                            if isinstance(result, dict):
                                state.update(result)
                        except Exception as e:
                            print(f"   ⚠️ Query interface skipped: {e}")
                            state["errors"].append(f"Query interface failed: {e}")

        else:
            # Forward engineering workflow
            print("📋 Starting forward engineering from requirements...")

            # Requirements
            state["current_step"] = "requirements"
            result = self.agents["requirements"].execute(state)
            if isinstance(result, dict):
                state.update(result)

            # Conceptual model
            if state.get("requirements"):
                print("\n📐 Creating conceptual model...")
                state["current_step"] = "conceptual"
                result = self.agents["conceptual"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

            # Logical model
            if state.get("conceptual_model"):
                print("\n🗂️ Creating logical model...")
                state["current_step"] = "logical"
                result = self.agents["logical"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

            # Physical model
            if state.get("logical_model"):
                print("\n💾 Creating physical model...")
                state["current_step"] = "physical"
                result = self.agents["physical"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

            # Implementation
            if state.get("physical_model"):
                print("\n🔨 Implementing database...")
                state["current_step"] = "implementation"
                result = self.agents["implementation"].execute(state)
                if isinstance(result, dict):
                    state.update(result)

            # ADD THIS - Data Products
            if state.get("physical_model"):
                print("\n📡 Generating data products...")
                state["current_step"] = "data_product"

                # Ensure sql_ddl is available
                if "implementation_artifacts" not in state:
                    state["implementation_artifacts"] = {}
                if "sql_ddl" not in state["implementation_artifacts"]:
                    state["implementation_artifacts"]["sql_ddl"] = state.get("sql_ddl")

                try:
                    result = self.agents["data_product"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)
                except Exception as e:
                    print(f"   ⚠️ Data products skipped: {e}")
                    state["errors"].append(f"Data products failed: {e}")

            # Visualization
            if state.get("database_path"):
                print(f"\n📈 Creating visualizations...")
                print(f"   Database path: {state.get('database_path')}")
                state["current_step"] = "visualization"
                try:
                    result = self.agents["visualization"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)
                except Exception as e:
                    print(f"   ⚠️ Visualization skipped: {e}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    state["errors"].append(f"Visualization failed: {e}")
            else:
                print(f"\n⚠️ Skipping visualizations - no database_path in state")
                print(f"   Available keys: {list(state.keys())}")

            # Query interface
            if state.get("database_path"):
                print("\n💬 Setting up query interface...")
                state["current_step"] = "conversational_query"
                try:
                    result = self.agents["conversational_query"].execute(state)
                    if isinstance(result, dict):
                        state.update(result)
                except Exception as e:
                    print(f"   ⚠️ Query interface skipped: {e}")
                    state["errors"].append(f"Query interface failed: {e}")

        # Final summary
        state["final_step"] = state["current_step"]
        
        # Create downloadable ZIP of all artifacts
        if self.artifact_manager:
            try:
                import zipfile
                import os
                from pathlib import Path
                
                project_path = self.artifact_manager.project_path
                zip_filename = f"{project_id}_artifacts.zip"
                zip_path = project_path.parent / zip_filename
                
                print(f"\n📦 Creating artifact package: {zip_filename}")
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through project directory
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(project_path.parent)
                            zipf.write(file_path, arcname)
                
                state["artifacts_zip"] = str(zip_path)
                state["artifacts_zip_name"] = zip_filename
                print(f"   ✅ Artifact package created: {zip_filename}")
                
                # Create artifact manifest for UI
                artifacts_manifest = {
                    "conceptual_model": str(project_path / "02_models" / "conceptual_model.md"),
                    "logical_model": str(project_path / "02_models" / "logical_model.md"),
                    "physical_ddl": str(project_path / "03_implementation" / "database_schema.sql"),
                    "glossary": str(project_path / "04_glossary" / "glossary.md"),
                    "data_products": str(project_path / "05_data_products"),
                    "visualizations": str(project_path / "08_visualizations"),
                    "table_viewer": str(project_path / "08_visualizations" / "table_viewer.html"),
                    "data_explorer": str(project_path / "08_visualizations" / "data_explorer.html"),
                    "database": state.get("database_path"),
                    "readme": str(project_path / "README.md"),
                    "full_package": str(zip_path)
                }
                state["artifacts_manifest"] = artifacts_manifest
                
            except Exception as e:
                print(f"   ⚠️ Could not create ZIP: {e}")
                state.setdefault("errors", []).append(f"ZIP creation failed: {e}")

        print(f"\n{'='*60}")
        print("✅ Workflow Complete!")
        print(f"📊 Final Step: {state['final_step']}")
        if state.get("database_path"):
            print(f"📁 Database: {state['database_path']}")
        if state.get("artifacts_zip_name"):
            print(f"📦 Download package: {state['artifacts_zip_name']}")
        if state.get("errors"):
            print(f"⚠️ Warnings: {', '.join(state['errors'])}")
        print(f"{'='*60}\n")

        return state

class SchemaAwareRawDataAgent:
    """
    Original Raw Data Agent that works with the Inventory Management demo
    Gets data from state during execute rather than at initialization
    """

    def __init__(self, llm, artifact_manager):
        """Original initialization - just LLM and artifact manager"""
        self.llm = llm
        self.artifact_manager = artifact_manager

    def execute(self, state):
        """
        Execute raw data transformation
        Gets DDL and raw data from state
        """
        print("📊 Raw Data Agent starting...")

        # Get data from state
        ddl_content = state.get("ddl_content", "")
        raw_data_content = state.get("raw_data_content")
        mapping_instructions = state.get("mapping_instructions", "")

        # Skip if no raw data provided
        if not raw_data_content:
            print("   No raw data provided, skipping...")
            return state

        try:
            # Parse raw data
            import pandas as pd
            from io import BytesIO, StringIO

            # Try to determine file type and parse
            if isinstance(raw_data_content, pd.DataFrame):
                # Already a DataFrame
                raw_data = raw_data_content
            elif isinstance(raw_data_content, dict):
                # Dictionary of DataFrames
                raw_data = raw_data_content
            else:
                # Try parsing as CSV or Excel
                try:
                    if isinstance(raw_data_content, bytes):
                        content_str = raw_data_content.decode('utf-8')
                    else:
                        content_str = raw_data_content
                    raw_data = pd.read_csv(StringIO(content_str))
                except:
                    try:
                        raw_data = pd.read_excel(BytesIO(raw_data_content))
                    except:
                        print("   Error: Could not parse raw data file")
                        state["errors"] = state.get("errors", []) + ["Could not parse raw data"]
                        return state

            # For now, just save the raw data as CSV
            # (You can add more sophisticated mapping logic here if needed)
            if self.artifact_manager:
                csv_dir = self.artifact_manager.project_path / "01_data"
                csv_dir.mkdir(parents=True, exist_ok=True)

                if isinstance(raw_data, pd.DataFrame):
                    # Single DataFrame
                    csv_path = csv_dir / "raw_data.csv"
                    raw_data.to_csv(csv_path, index=False)
                    state["csv_files"] = [str(csv_path)]
                    print(f"   Saved raw data to {csv_path}")
                elif isinstance(raw_data, dict):
                    # Multiple DataFrames
                    csv_files = []
                    for name, df in raw_data.items():
                        csv_path = csv_dir / f"{name}.csv"
                        df.to_csv(csv_path, index=False)
                        csv_files.append(str(csv_path))
                        print(f"   Saved {name} to {csv_path}")
                    state["csv_files"] = csv_files

            state["csv_generation_complete"] = True
            print("   ✅ Raw data processing complete")

        except Exception as e:
            print(f"   Error processing raw data: {e}")
            state["errors"] = state.get("errors", []) + [f"Raw data processing failed: {e}"]
            state["csv_generation_complete"] = False

        return state

class LabOfTheFutureOrchestrator:
    """
    Specialized orchestrator for Lab of the Future workflow
    DDL → Data + Mapping → Database → Products/Viz/Query
    """

    def __init__(self, llm, project_config: ProjectConfig):
        self.llm = llm
        self.project_config = project_config
        self.artifact_manager = SimpleArtifactManager(project_config)

        # Store references but don't create agents yet
        # (Raw data agent needs data that's not available at init time)
        self.agents = {}

    def execute_lab_workflow(self, project_id: str, ddl_content: str,
                             raw_data_content: bytes, mapping_instructions: str):
        """Execute the Lab of the Future workflow"""

        print("\nLab of the Future Workflow Starting...")
        print("="*60)

        state = {
            "project_id": project_id,
            "ddl_content": ddl_content,
            "raw_data_content": raw_data_content,
            "mapping_instructions": mapping_instructions,
            "errors": [],
            "current_step": "starting"
        }

        # Step 1: Store DDL as physical model
        print("1/6 - Storing DDL as Physical Model...")
        state["physical_model"] = ddl_content
        state["sql_ddl"] = ddl_content
        if self.artifact_manager:
            ddl_file = self.artifact_manager.project_path / "02_models" / "physical_model.sql"
            ddl_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ddl_file, 'w', encoding='utf-8') as f:
                f.write(ddl_content)
        print("   Done.")

        # Step 2: Transform raw data to CSV using mapping
        print("2/6 - Transforming Raw Data to CSV...")
        print("LLM-Guided Schema-Aware Raw Data Agent Starting...")

        try:
            # Parse the raw data first
            import pandas as pd
            from io import BytesIO, StringIO

            # Determine file type and parse
            raw_data = None
            file_name = "unknown_file"

            # Try to parse as CSV first
            try:
                if isinstance(raw_data_content, bytes):
                    content_str = raw_data_content.decode('utf-8')
                else:
                    content_str = raw_data_content
                raw_data = pd.read_csv(StringIO(content_str))
                file_type = "csv"
                print(f"   Detected file type: csv")
            except:
                # Try Excel
                try:
                    raw_data = pd.read_excel(BytesIO(raw_data_content))
                    file_type = "excel"
                    print(f"   Detected file type: excel")
                except:
                    print("   Error: Could not parse raw data file")
                    state["errors"].append("Could not parse raw data file")
                    return state

            # Now create the agent with all required parameters
            #from your_module import SchemaAwareRawDataAgent  # Add correct import

            # Prepare raw_data as dict (sheet_name -> dataframe)
            raw_data_dict = {file_name: raw_data}

            # Create agent with all required arguments
            raw_data_agent = SchemaAwareRawDataAgent(
                ddl_content=ddl_content,
                raw_data=raw_data_dict,
                mapping_instructions=mapping_instructions,
                llm=self.llm,
                logger=logging.getLogger(__name__)
            )

            # Transform the data
            csv_data = raw_data_agent.transform()

            # Save CSV files
            if self.artifact_manager:
                csv_dir = self.artifact_manager.project_path / "01_data"
                csv_dir.mkdir(parents=True, exist_ok=True)

                for filename, df in csv_data.items():
                    csv_path = csv_dir / filename
                    df.to_csv(csv_path, index=False)
                    print(f"   Saved: {filename}")

                state["csv_files"] = list(csv_data.keys())
                state["csv_generation_complete"] = True

            print("   Done.")

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            state["errors"].append(f"Raw data transformation failed: {e}")
            return state

        # Step 3: Provision database from DDL
        print("3/6 - Database Provisioning...")
        try:
            # Create implementation agent
            #from your_module import EnhancedImplementationAgent  # Add correct import
            implementation_agent = EnhancedImplementationAgent(self.llm, self.artifact_manager)

            # Prepare state for implementation
            impl_state = {
                "sql_ddl": ddl_content,
                "csv_files": state.get("csv_files", []),
                "project_id": project_id
            }

            # Execute database provisioning
            result = implementation_agent.execute(impl_state)

            if result.get("database_provisioned"):
                state["database_provisioned"] = True
                state["database_path"] = result.get("database_path")
                print(f"   Database created: {result.get('database_path')}")
            else:
                print("   Warning: Database provisioning incomplete")
                state["errors"].append("Database provisioning failed")

        except Exception as e:
            print(f"   Error: {e}")
            state["errors"].append(f"Database provisioning failed: {e}")

        print("   Done.")

        # Step 4: Data Products
        print("4/6 - Generating Data Products...")
        try:
            #from your_module import DataProductAgent  # Add correct import
            data_product_agent = DataProductAgent(self.llm, self.artifact_manager)
            dp_result = data_product_agent.execute(state)
            state["data_products_generated"] = dp_result.get("data_products_generated", False)
        except Exception as e:
            print(f"   Warning: Data product generation failed: {e}")
            state["errors"].append(f"Data product generation failed: {e}")
        print("   Done.")

        # Step 5: Data Visualization
        print("5/6 - Data Visualization...")
        print("   📊 Creating Data Visualizations...")

        if state.get("database_path"):
            try:
                #from your_module import VisualizationAgent  # Add correct import
                viz_agent = VisualizationAgent(self.llm, self.artifact_manager)
                viz_result = viz_agent.execute(state)
                state["visualizations_created"] = viz_result.get("visualizations_created", False)
            except Exception as e:
                print(f"   Warning: Visualization failed: {e}")
                state["errors"].append(f"Visualization failed: {e}")
        else:
            print("   ❌ Could not find database")
            state["errors"].append("No database found for visualization")

        print("   ✅ Done.")

        # Step 6: Natural Language Query Interface
        print("6/6 - Natural Language Query Interface...")
        print("   💬 Initializing Conversational Query Interface...")

        if state.get("database_path"):
            try:
                #from your_module import ConversationalQueryAgent  # Add correct import
                query_agent = ConversationalQueryAgent(self.llm, self.artifact_manager)
                query_result = query_agent.execute(state)
                state["conversational_query_ready"] = query_result.get("query_interface_ready", False)
            except Exception as e:
                print(f"   Warning: Query interface setup failed: {e}")
                state["errors"].append(f"Query interface setup failed: {e}")
        else:
            print("   ❌ No database found")
            state["errors"].append("No database found for querying")

        print("   ✅ Done.")

        # Complete
        print("\n" + "="*60)
        print("✅ LAB OF THE FUTURE WORKFLOW COMPLETED!")
        print(f"📋 Project ID: {project_id}")
        print(f"📊 Status: {'completed' if not state['errors'] else 'completed with warnings'}")

        if state['errors']:
            print(f"⚠️ Warnings/Errors: {state['errors']}")

        if state.get("database_path"):
            print(f"💾 Database: {state['database_path']}")

        print("="*60)

        return state

class DataArchitectAPI:
    """Main API for the Data Architect system"""

    def __init__(self, llm, base_path: str = "./projects"):
        self.llm = llm
        self.base_path = base_path
        self.orchestrators = {}

    def create_project(self, project_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new data architecture project"""
        project_id = project_request.get("project_id", "project_" + str(abs(hash(str(project_request)))))

        # Create project configuration
        project_config = ProjectConfig(
            project_id=project_id,
            base_path=self.base_path,
            create_zip=project_request.get("create_zip", True)
        )

        # Get custom temperature settings if provided
        custom_temps = project_request.get("agent_temperatures")
        if custom_temps:
            global AGENT_TEMPERATURES
            for agent, temp in custom_temps.items():
                AGENT_TEMPERATURES[agent] = temp

        # Create orchestrator
        orchestrator = DataArchitectOrchestrator(self.llm, project_config)
        self.orchestrators[project_id] = orchestrator

        # Convert request to messages
        messages = []
        if project_request.get("description"):
            messages.append(type('Message', (), {"content": project_request["description"]})())

        if project_request.get("requirements"):
            messages.append(type('Message', (), {"content": "Requirements: " + project_request["requirements"]})())

        # CRITICAL: If explicit_entities provided, create initial state with them
        initial_state = None
        if project_request.get("explicit_entities"):
            entities = project_request["explicit_entities"]
            print(f"\n🎯 Using explicit entities from chatbot: {', '.join(entities)}")
            initial_state = {
                "project_id": project_id,
                "inferred_entities": entities,  # This will be preserved through workflow
                "messages": messages,
                "current_step": "starting",
                "errors": []
            }

        # Execute workflow
        if initial_state:
            result = orchestrator.execute_workflow(project_id, messages, initial_state=initial_state)
        else:
            result = orchestrator.execute_workflow(project_id, messages)

        # Prepare response
        response = {
            "project_id": project_id,
            "status": "created",
            "success": True,
            "current_step": result.get("current_step", "unknown"),
            "errors": result.get("errors", [])
        }

        # Add file information
        if orchestrator.artifact_manager:
            if hasattr(orchestrator.artifact_manager, 'created_files'):
                response["files_created"] = len(orchestrator.artifact_manager.created_files)
            response["project_path"] = str(orchestrator.artifact_manager.project_path)
        
        # Add artifact manifest and ZIP download
        if result.get("artifacts_manifest"):
            response["artifacts"] = result["artifacts_manifest"]
        if result.get("artifacts_zip_name"):
            response["zip_file"] = result["artifacts_zip_name"]
            response["zip_path"] = result.get("artifacts_zip")

        return response

    def create_project_from_csvs(
        self,
        project_request: Dict[str, Any],
        csv_paths: List[str],
        db_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a project seeded by CSV → schema inference"""
        project_id = project_request.get("project_id") or f"csv_project_{len(self.orchestrators)+1}"

        # Create project config
        project_config = ProjectConfig(
            project_id=project_id,
            base_path=self.base_path,
            create_zip=project_request.get("create_zip", True)
        )

        # Create orchestrator
        orchestrator = DataArchitectOrchestrator(self.llm, project_config)
        self.orchestrators[project_id] = orchestrator

        # Convert request to messages
        messages = []
        if project_request.get("description"):
            messages.append(type('Message', (), {"content": project_request["description"]})())

        # Run workflow with CSVs - ALWAYS reverse engineering from CSV
        result = orchestrator.execute_workflow(
            project_id,
            messages,
            csv_files=csv_paths,
            db_url=db_url,
            reverse=True  # CSV upload is ALWAYS reverse engineering
        )

        response = {
            "project_id": project_id,
            "status": "created",
            "current_step": result.get("current_step", "unknown"),
            "errors": result.get("errors", [])
        }

        if orchestrator.artifact_manager:
            response["project_path"] = str(orchestrator.artifact_manager.project_path)
            if hasattr(orchestrator.artifact_manager, 'created_files'):
                response["files_created"] = len(orchestrator.artifact_manager.created_files)
            
            # Add artifact manifest for visualization buttons
            project_path = orchestrator.artifact_manager.project_path
            artifacts_manifest = {
                "conceptual_model": str(project_path / "02_models" / "conceptual_model.md"),
                "logical_model": str(project_path / "02_models" / "logical_model.md"),
                "physical_ddl": str(project_path / "03_implementation" / "database_schema.sql"),
                "glossary": str(project_path / "04_glossary" / "glossary.md"),
                "table_viewer": str(project_path / "08_visualizations" / "table_viewer.html"),
                "data_explorer": str(project_path / "08_visualizations" / "data_explorer.html"),
            }
            response["artifacts"] = artifacts_manifest
        
        # Add database path if available
        if result.get("database_path"):
            response["database_path"] = result["database_path"]
        
        # Add DDL content for query interface
        if result.get("ddl_content"):
            response["ddl_content"] = result["ddl_content"]
        elif result.get("physical_model"):
            response["ddl_content"] = result["physical_model"]
        
        # Add zip file info if available
        if result.get("artifacts_zip_name"):
            response["zip_file"] = result["artifacts_zip_name"]
            response["zip_path"] = result.get("artifacts_zip")

        return response

    def create_project_from_ddl(self, project_request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a project by reverse engineering from DDL"""
        project_id = project_request.get("project_id", "reverse_eng_" + str(abs(hash(str(project_request)))))
        ddl_content = project_request.get("ddl_content", "")

        if not ddl_content:
            return {"error": "No DDL content provided"}

        # Create project configuration
        project_config = ProjectConfig(
            project_id=project_id,
            base_path=self.base_path,
            create_zip=project_request.get("create_zip", True)
        )

        # Create orchestrator
        orchestrator = DataArchitectOrchestrator(self.llm, project_config)
        orchestrator.agents["ddl_reverse_engineering"] = SimplifiedDDLReverseEngineeringAgent(
            self.llm, orchestrator.artifact_manager
        )
        self.orchestrators[project_id] = orchestrator

        # Create initial state with DDL content
        state = {
            "project_id": project_id,
            "ddl_content": ddl_content,
            "requirements": None,
            "conceptual_model": None,
            "logical_model": None,
            "physical_model": None,
            "implementation_artifacts": None,
            "test_results": None,
            "data_product_config": None,
            "messages": [],
            "current_step": "starting",
            "errors": [],
            "reverse_engineered": True
        }

        try:
            # Execute reverse engineering
            state = orchestrator.agents["ddl_reverse_engineering"].execute(state)

            if state.get("errors"):
                return {
                    "project_id": project_id,
                    "status": "error",
                    "errors": state.get("errors", [])
                }

            # Execute remaining agents
            remaining_agents = [
                ("implementation", "Database Implementation"),  # NEW!
                ("glossary", "Domain Glossary"),
                ("data_product", "Data Product APIs"),
                ("ontology", "Ontology Models")
            ]

            for step, (agent_name, description) in enumerate(remaining_agents, 2):
                agent_instance = orchestrator.agents.get(agent_name)
                if agent_instance:
                    try:
                        result = agent_instance.execute(state)
                        if isinstance(result, dict):
                            state.update(result)
                    except Exception as e:
                        # Don't fail the whole workflow - just log the error
                        error_msg = f"{description} failed: {str(e)}"
                        print(f"   ⚠️ {error_msg}")
                        if "errors" not in state:
                            state["errors"] = []
                        state["errors"].append(error_msg)
            
            # NEW: Add Visualization Agent (after implementation creates database)
            if state.get("database_path"):
                print("\n📈 Creating visualizations...")
                try:
                    viz_result = orchestrator.agents["visualization"].execute(state)
                    if isinstance(viz_result, dict):
                        state.update(viz_result)
                    print("   ✅ Visualizations created")
                except Exception as e:
                    print(f"   ⚠️ Visualization skipped: {e}")
                    if "errors" not in state:
                        state["errors"] = []
                    state["errors"].append(f"Visualization failed: {e}")
            
            # NEW: Add Conversational Query Interface
            if state.get("database_path"):
                print("\n💬 Setting up query interface...")
                try:
                    query_result = orchestrator.agents["conversational_query"].execute(state)
                    if isinstance(query_result, dict):
                        state.update(query_result)
                    print("   ✅ Query interface ready")
                except Exception as e:
                    print(f"   ⚠️ Query interface skipped: {e}")
                    if "errors" not in state:
                        state["errors"] = []
                    state["errors"].append(f"Query interface failed: {e}")

            # Create ZIP and artifacts even if some agents failed
            if orchestrator.artifact_manager:
                # Try to create artifacts from what we have
                try:
                    # Create ZIP
                    import zipfile
                    import os
                    from pathlib import Path
                    
                    project_path = orchestrator.artifact_manager.project_path
                    zip_filename = f"{project_id}_artifacts.zip"
                    zip_path = project_path.parent / zip_filename
                    
                    print(f"\n📦 Creating artifact package: {zip_filename}")
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(project_path):
                            for file in files:
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(project_path.parent)
                                zipf.write(file_path, arcname)
                    
                    state["artifacts_zip"] = str(zip_path)
                    state["artifacts_zip_name"] = zip_filename
                    print(f"   ✅ Artifact package created: {zip_filename}")
                except Exception as e:
                    print(f"   ⚠️ Could not create ZIP: {e}")
                    if "errors" not in state:
                        state["errors"] = []
                    state["errors"].append(f"ZIP creation failed: {e}")
                
                # Generate README
                if not state.get("errors") or len(state.get("errors", [])) < 3:
                    try:
                        orchestrator.artifact_manager.generate_readme()
                    except Exception as e:
                        print(f"   ⚠️ README generation failed: {e}")

            # Prepare response
            response = {
                "project_id": project_id,
                "status": "reverse_engineered",
                "success": True,  # Mark as success even with warnings
                "current_step": state.get("current_step", "unknown"),
                "errors": state.get("errors", [])
            }

            if orchestrator.artifact_manager:
                if hasattr(orchestrator.artifact_manager, 'created_files'):
                    response["files_created"] = len(orchestrator.artifact_manager.created_files)
                response["project_path"] = str(orchestrator.artifact_manager.project_path)
            
            # Add artifact info
            if state.get("artifacts_zip_name"):
                response["zip_file"] = state["artifacts_zip_name"]
                response["zip_path"] = state.get("artifacts_zip")
            
            # Add artifact manifest
            if orchestrator.artifact_manager:
                project_path = orchestrator.artifact_manager.project_path
                artifacts_manifest = {
                    "conceptual_model": str(project_path / "02_models" / "conceptual_model.md"),
                    "logical_model": str(project_path / "02_models" / "logical_model.md"),
                    "physical_ddl": str(project_path / "03_implementation" / "database_schema.sql"),
                    "glossary": str(project_path / "04_glossary" / "glossary.md"),
                    "table_viewer": str(project_path / "08_visualizations" / "table_viewer.html"),
                    "data_explorer": str(project_path / "08_visualizations" / "data_explorer.html"),
                }
                response["artifacts"] = artifacts_manifest

            return response

        except Exception as e:
            return {
                "project_id": project_id,
                "status": "error",
                "errors": [str(e)]
            }

    def create_lab_project(self, project_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Lab of the Future project

        Args:
            project_request: Must contain:
                - ddl_content: SQL DDL
                - raw_data_content: Raw data bytes
                - mapping_instructions: How to map data
                - project_id (optional)
        """
        project_id = project_request.get("project_id", f"lab_{len(self.orchestrators)+1}")

        ddl_content = project_request.get("ddl_content", "")
        raw_data_content = project_request.get("raw_data_content")
        mapping_instructions = project_request.get("mapping_instructions", "")

        if not ddl_content:
            return {"error": "No DDL content provided", "status": "error"}
        if not raw_data_content:
            return {"error": "No raw data provided", "status": "error"}
        if not mapping_instructions:
            return {"error": "No mapping instructions provided", "status": "error"}

        # Create project config
        project_config = ProjectConfig(
            project_id=project_id,
            base_path=self.base_path,
            create_zip=project_request.get("create_zip", True)
        )

        # Create Lab orchestrator
        lab_orchestrator = LabOfTheFutureOrchestrator(self.llm, project_config)
        self.orchestrators[project_id] = lab_orchestrator

        # Execute Lab workflow
        result = lab_orchestrator.execute_lab_workflow(
            project_id,
            ddl_content,
            raw_data_content,
            mapping_instructions
        )

        # Prepare response
        response = {
            "project_id": project_id,
            "status": "completed" if not result.get("errors") else "completed_with_warnings",
            "workflow_type": "lab_of_future",
            "database_path": result.get("database_path"),
            "database_provisioned": result.get("database_provisioned", False),
            "errors": result.get("errors", []),
            "query_interface_ready": result.get("conversational_query_ready", False)
        }

        if lab_orchestrator.artifact_manager:
            response["project_path"] = str(lab_orchestrator.artifact_manager.project_path)

        return response

    def create_project_from_ddl_with_csv(self, project_request: dict) -> dict:
        """
        Create a project from DDL with CSV data files
        Combines DDL schema with CSV data for a complete database
        """
        project_id = f"ddl_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.orchestrators)+1}"

        ddl_content = project_request.get("ddl_content", "")
        csv_files = project_request.get("csv_files", {})
        domain = project_request.get("domain", "labs")

        if not ddl_content:
            return {"error": "No DDL content provided", "status": "error"}

        if not csv_files:
            return {"error": "No CSV files provided", "status": "error"}

        # Create project config
        project_config = ProjectConfig(
            project_id=project_id,
            base_path=self.base_path,
            create_zip=project_request.get("create_zip", True)
        )

        # Create orchestrator
        orchestrator = DataArchitectOrchestrator(self.llm, project_config)
        self.orchestrators[project_id] = orchestrator

        # Save the CSV files to the project directory
        if orchestrator.artifact_manager:
            data_dir = orchestrator.artifact_manager.project_path / "01_data"
            data_dir.mkdir(parents=True, exist_ok=True)

            csv_file_paths = []
            for filename, content in csv_files.items():
                file_path = data_dir / filename

                # Handle different content types
                if isinstance(content, pd.DataFrame):
                    content.to_csv(file_path, index=False)
                elif isinstance(content, str):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    # Assume it's bytes
                    with open(file_path, 'wb') as f:
                        f.write(content)

                csv_file_paths.append(str(file_path))
                print(f"   Saved CSV: {filename}")

        # Execute the workflow with DDL and CSV files
        result = orchestrator.execute_workflow(
            project_id=project_id,
            ddl_content=ddl_content,
            csv_files=csv_file_paths,  # Pass the file paths
            reverse=True  # This is reverse engineering from DDL+CSV
        )

        # Prepare response
        response = {
            "project_id": project_id,
            "status": "completed" if result.get("final_step") else "incomplete",
            "workflow_type": "ddl_plus_csv_reverse",
            "domain": domain,
            "csv_files_count": len(csv_files),
            "database_provisioned": result.get("database_provisioned", False),
            "final_step": result.get("final_step"),
            "errors": result.get("errors", [])
        }

        if orchestrator.artifact_manager:
            response["project_path"] = str(orchestrator.artifact_manager.project_path)

        return response




