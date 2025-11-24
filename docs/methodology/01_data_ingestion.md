# Methodology: Data Ingestion and Architecture

**Version:** 1.0.0  
**Module:** Data Engineering  
**Technical Stack:** Python 3.12, SQLAlchemy 2.0, PostgreSQL 16, Pydantic

---

## 1. Abstract

The Data Ingestion module serves as the foundational layer of the Opponent-Adjusted Metrics system. Its primary objective is to transform semi-structured, hierarchical event data (JSON) provided by StatsBomb into a normalized, relational structure (PostgreSQL) suitable for high-performance analytical querying and machine learning feature extraction. This document details the architectural decisions, the Extract-Transform-Load (ETL) logic, and the schema design principles that ensure data integrity, idempotency, and scalability.

## 2. Source Data Characteristics

The system consumes **StatsBomb Open Data**, which presents unique challenges typical of modern sports telemetry:

1.  **Hierarchical Depth:** A single match event is not a flat record. It contains nested objects for `tactics` (lineups), `location` (coordinates), `shot` (outcome, technique, body part), and crucially, the `freeze_frame`—a snapshot of all player positions at the moment of a shot.
2.  **Event Polymorphism:** The schema is polymorphic. A "Pass" event has attributes like `length` and `angle`, while a "Duel" event has `outcome` and `type`.
3.  **Relational Dependencies:** Events reference entities (Players, Teams, Competitions) that must be resolved to consistent foreign keys.

## 3. Database Schema Design

To address these characteristics, we implemented a hybrid relational schema using **SQLAlchemy ORM**. The design balances strict normalization (for entity integrity) with JSONB flexibility (for evolving event attributes).

### 3.1 Entity-Relationship Model

The schema follows a modified Star Schema approach:

*   **Dimension Tables:**
    *   `competitions`: Stores tournament metadata (e.g., "Premier League 2015/16").
    *   `teams`: Unique registry of football clubs/nations.
    *   `players`: Unique registry of athletes.
    *   `matches`: The central fact table for fixture metadata, linking dimensions.

*   **Fact Tables:**
    *   `events`: The granular atomic unit of the game.
    *   `possessions`: An aggregated fact table representing continuous phases of play.

### 3.2 The `events` Table Architecture

The `events` table is the system's workhorse. To handle the polymorphic nature of football events without creating hundreds of sparse columns, we utilized PostgreSQL's `JSONB` data type.

```python
class Event(Base):
    __tablename__ = "events"

    # Primary Keys and Foreign Keys
    id = Column(String, primary_key=True)  # UUID from StatsBomb
    match_id = Column(Integer, ForeignKey("matches.id"), index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), index=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)

    # Common Attributes (First-Class Columns)
    period = Column(Integer)
    timestamp = Column(Time)
    minute = Column(Integer)
    second = Column(Integer)
    type = Column(String, index=True)  # e.g., "Pass", "Shot"
    location_x = Column(Float)
    location_y = Column(Float)

    # Polymorphic Attributes (JSONB)
    extra_attributes = Column(JSONB)
```

**Design Rationale:**
*   **Indexing Strategy:** High-cardinality columns used for filtering (`match_id`, `type`, `player_id`) are B-Tree indexed.
*   **Hybrid Storage:** "First-class" columns are used for attributes common to >80% of events (location, time). Rare or specific attributes (e.g., `pass.switch`, `foul_committed.card`) are stored in `extra_attributes`. This prevents schema bloat while maintaining queryability via JSON operators (e.g., `extra_attributes ->> 'pass_switch'`).

### 3.3 The `shots` View/Table

Given the project's focus on Expected Goals (xG), shots require special handling. We implemented a specialized `Shot` model that flattens the nested `shot` dictionary from the raw event.

*   **Freeze Frame Parsing:** The `shot.freeze_frame` is a list of dictionaries. During ingestion, this is parsed to extract derived metrics *at the point of load*, such as:
    *   `goalkeeper_x`, `goalkeeper_y`
    *   `defender_count_in_cone`
*   **Outcome Mapping:** StatsBomb outcomes (e.g., "Goal", "Saved", "Off T") are mapped to a binary `is_goal` flag and a categorical `outcome` enum.

## 4. ETL Pipeline Architecture

The ingestion pipeline is orchestrated via Python scripts (`scripts/ingest_*.py`), adhering to the following principles:

### 4.1 Idempotency and Upserts
Football data is often restated (e.g., a goal awarded to a different scorer post-match). The pipeline is strictly **idempotent**.
*   **Strategy:** We use the "Check-then-Insert" or "Upsert" (Update on Conflict) pattern.
*   **Implementation:** Before inserting a match, the system checks for its existence. If found, it can either skip or cascade-delete and re-ingest, ensuring the database never contains duplicate records for the same fixture.

### 4.2 Transaction Management
To ensure data consistency, ingestion operates within strict **ACID transactions**.
*   **Scope:** A single match is the unit of work.
*   **Rollback:** If *any* event within a match fails validation (e.g., missing player ID), the entire transaction is rolled back. This prevents "zombie matches" where only half the events are loaded.

### 4.3 Coordinate Normalization
StatsBomb uses a coordinate system of `(0,0)` (top-left) to `(120,80)` (bottom-right).
*   **Transformation:** All coordinates are validated to ensure they fall within these bounds.
*   **Orientation:** The pipeline ensures that the attacking direction is consistent (left-to-right) for the team in possession, flipping coordinates if necessary based on the period and team side.

## 5. Implementation Details

### 5.1 The `ingest_events.py` Workflow

1.  **Discovery:** The script scans the local data directory for JSON files matching the requested Competition/Season.
2.  **Session Initialization:** A `SessionLocal` context manager is spawned.
3.  **Batch Processing:**
    *   Events are loaded into memory.
    *   **Entity Resolution:** New Players and Teams encountered in the match are identified. If they don't exist in the DB, they are staged for insertion.
    *   **Event Mapping:** Raw dictionaries are mapped to `Event` objects.
4.  **Bulk Persistence:**
    *   We utilize `session.bulk_save_objects()` for high-throughput insertion. This bypasses some ORM overhead, offering a ~10x speedup over individual `session.add()` calls.
    *   **Performance:** The system ingests a full Premier League season (~380 matches, ~1.2M events) in under 5 minutes on standard hardware.

### 5.2 Handling Freeze Frames
The `freeze_frame` is critical for the Contextual Model.
*   **Raw Format:** `[{'location': [x, y], 'player': {'id': ...}, 'teammate': False}, ...]`
*   **Ingestion Logic:**
    *   Filter for `teammate=False` (Opponents).
    *   Calculate Euclidean distance from each opponent to the shooter.
    *   Store the raw freeze frame as JSONB for future feature engineering, but also pre-calculate `gk_location` if a player is flagged as the goalkeeper.

## 6. Quality Assurance

Data integrity is enforced at multiple levels:
1.  **Database Constraints:** Foreign keys ensure no event references a non-existent player.
2.  **Pydantic Validation:** Input JSON is validated against strict schemas before touching the database.
3.  **Post-Ingestion Audits:** The `filter_report.json` artifact tracks how many events were dropped due to missing critical attributes (e.g., shots without location data), providing transparency into data quality.

## 7. Conclusion

The Data Ingestion module provides a robust, scalable foundation for the project. By normalizing the complex StatsBomb hierarchy into a query-optimized PostgreSQL schema, we enable rapid iteration in the subsequent Analysis and Modeling phases. The strict adherence to ACID principles and idempotency ensures that the analytical dataset is both reliable and reproducible.
