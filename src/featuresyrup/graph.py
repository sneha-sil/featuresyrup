from featuresyrup.classes import DictData, Node, Edge, GraphInfo, Targets
from .functions import *
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json

# Optional imports for visualization and ML frameworks
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import networkx as nx
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def json_wrapper(value):
    """Convert dict, list, or tuple to JSON string for database storage."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def scalar_wrapper(value):
    """Convert NumPy scalar values to native Python scalars for SQLite storage."""
    if isinstance(value, np.generic):
        return value.item()
    return value


CATALYST_NODE_COLUMNS = [
    "ACSF_values",
    "SOAP_values",
    "proton_isotropic",
    "proton_anisotropic",
    "carbon_isotropic",
    "carbon_anisotropic",
    "fukui_f_plus",
    "fukui_f_minus",
    "fukui_f_zero",
]

CATALYST_TARGET_COLUMNS = [
    "HOMO",
    "LUMO",
    "HOMO_LUMO_gap",
    "buried_volume",
    "fraction_vbur",
    "free_volume",
    "sasa_area",
    "sasa_volume",
    "IR_freq_mean",
    "IR_freq_max",
    "IR_freq_min",
    "IR_freq_std",
    "IR_freq_median",
    "IR_eps_mean",
    "IR_eps_max",
    "IR_eps_min",
    "IR_eps_std",
    "IR_eps_median",
    "IR_intensity_mean",
    "IR_intensity_max",
    "IR_intensity_min",
    "IR_intensity_std",
    "IR_intensity_median",
    "IR_t2_mean",
    "IR_t2_max",
    "IR_t2_min",
    "IR_t2_std",
    "IR_t2_median",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_magnitude",
    "isotropic_polarizability",
    "ionization_energy",
    "electron_affinity",
    "hardness",
    "chemical_potential",
    "electronegativity",
    "electrophilicity",
]

class MolecularGraph:
    
    def __init__(self, qm_data: DictData):
        """
        Initialize molecular graph from quantum mechanical data.
        
        Args:
            qm_data (DictData): QM data object
        """
        self._qm_data = qm_data
        self.graph_info = GraphInfo(qm_data)
        self.nodes = Node(qm_data) if qm_data.num_atoms else None
        self.edges = Edge(qm_data) if qm_data.num_atoms else None
        self.targets = Targets(qm_data)
        
        self._node_features = None
        self._edge_features = None
        self._graph_features = None
        self._target_features = None
        
        # Validate structure on creation
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate molecular graph structure consistency."""
        if not self.nodes:
            logger.warning(f"Graph {self.id} has no atoms")
            return
            
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()
        
        # Validate node consistency
        if node_features:
            first_keys = set(node_features[0].keys())
            for i, node in enumerate(node_features[1:], 1):
                if set(node.keys()) != first_keys:
                    missing = first_keys - set(node.keys())
                    for key in missing:
                        node[key] = None
                    logger.warning(f"Padded missing keys {missing} for node {i}")
        
        # Validate edge indices
        num_nodes = len(node_features) if node_features else 0
        for edge in edge_features:
            i, j = edge.get('atom_i', 0), edge.get('atom_j', 0)
            if not (0 <= i < num_nodes and 0 <= j < num_nodes):
                logger.warning(f"Invalid edge indices ({i}, {j}) for {num_nodes} nodes")
    
    @property
    def id(self) -> Optional[str]:
        """Graph identifier."""
        return self.graph_info.id
    
    @property 
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return self._qm_data.num_atoms or 0
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds in the molecule."""
        edge_features = self.get_edge_features()
        return len(edge_features) if edge_features else 0
    
    def get_node_features(self, refresh: bool = False) -> List[Dict]:
        """
        Get node features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            List[Dict]: List of node feature dictionaries
        """
        if self._node_features is None or refresh:
            self._node_features = self.nodes.get_node_features() if self.nodes else []
        return self._node_features
    
    def get_node_features_ML(self) -> List[Dict]:
        """
        Get node features suitable for machine learning.
        
        Returns:
            List[Dict]: List of node feature dictionaries for ML
        """
        if not self.nodes:
            return []
        return self.nodes.get_node_features_ML()
    
    def get_edge_features(self, refresh: bool = False) -> List[Dict]:
        """
        Get edge features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            List[Dict]: List of edge feature dictionaries
        """
        if self._edge_features is None or refresh:
            self._edge_features = self.edges.get_edge_features() if self.edges else []
        return self._edge_features
    
    def get_edge_features_ML(self) -> List[Dict]:
        """
        Get edge features suitable for machine learning.
        
        Returns:
            List[Dict]: List of edge feature dictionaries for ML
        """
        if not self.edges:
            return []
        return self.edges.get_edge_features_ML()
    
    def get_graph_features(self, refresh: bool = False) -> Dict:
        """
        Get graph-level features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            Dict: Graph feature dictionary
        """
        if self._graph_features is None or refresh:
            self._graph_features = self.graph_info.get_graph_features()
        return self._graph_features
    
    def get_graph_features_ML(self) -> Dict:
        """
        Get graph-level features suitable for machine learning.
        
        Returns:
            Dict: Graph feature dictionary for ML
        """
        if not self.graph_info:
            return {}
        return self.graph_info.get_graph_features_ML()
    
    def get_target_features(self, refresh: bool = False) -> Dict:
        """
        Get target features with caching.
        
        Args:
            refresh (bool): Force refresh of cached data
            
        Returns:
            Dict: Target feature dictionary
        """
        if self._target_features is None or refresh:
            self._target_features = self.targets.get_targets()
        return self._target_features
    
    def get_target_features_ML(self) -> Dict:
        """
        Get target features suitable for machine learning.
        
        Returns:
            Dict: Target feature dictionary for ML
        """
        if not self.targets:
            return {}
        return self.targets.get_targets_ML()
    
    def get_molecular_data(self) -> Dict:
        """
        Get complete molecular data structure.
        
        Returns:
            Dict: Complete molecular data with graph, nodes, edges, and targets
        """
        return {
            'graph_id': self.id,
            'graph_info': self.get_graph_features(),
            'nodes': self.get_node_features(), 
            'edges': self.get_edge_features(),
            'targets': self.get_target_features(),
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
        }
    
    def get_ML_features(self) -> Dict:
        """
        Produces features suitable for machine learning tasks.
        
        Returns:
            Dict: dictionary of quantitative features.
        """
        return {
            'graph_id': self.id,
            'graph_info': self.get_graph_features_ML(),
            'nodes': self.get_node_features_ML(),
            'edges': self.get_edge_features_ML(),
            'targets': self.get_target_features_ML(),
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds
        }
        
    @classmethod
    def create_database(cls, db_path: Union[str, Path]) -> None:
        """
        Create SQLite database with graph-type dependent schema for molecular graphs.
        
        Args:
            db_path: Path to SQLite database file
        """
        db_path = Path(db_path)
        if not db_path.name.endswith('.db'):
            db_path = db_path.with_suffix('.db')
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Graphs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graphs (
                graph_id TEXT PRIMARY KEY,
                smiles TEXT,
                formula TEXT,
                molecular_mass REAL,
                num_atoms INTEGER,
                num_electrons INTEGER,
                charge REAL,
                num_bonds INTEGER,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Nodes table
        base_node_columns = '''
            node_id TEXT,
            graph_id TEXT,
            atom_index INTEGER,
            atomic_number INTEGER,
            atom_label TEXT,
            x_position REAL,
            y_position REAL, 
            z_position REAL,
            atomic_mass REAL,
            electronegativity REAL,
            covalent_radius REAL
        '''
        
        npa_nbo_common_columns = '''
            wiberg_bond_order_total REAL,
            bound_hydrogens INTEGER,
            node_degree INTEGER
        '''

        npa_specific_columns = '''
            electron_population REAL,
            nmb_population REAL,
            npa_charge REAL
        '''

        nbo_specific_columns = '''
            natural_charge REAL,
            core_population REAL,
            valence_population REAL,
            rydberg_population REAL,
            total_population REAL,
            core_orbital_occupancy REAL,
            core_orbital_energy REAL,
            lone_pair_1_occupancy REAL,
            lone_pair_1_energy REAL,
            lone_pair_2_occupancy REAL,
            lone_pair_2_energy REAL
        '''

        catalyst_node_columns = '''
            is_carbene_center INTEGER,
            ACSF_values TEXT,
            SOAP_values TEXT,
            proton_isotropic REAL,
            proton_anisotropic REAL,
            carbon_isotropic REAL,
            carbon_anisotropic REAL,
            fukui_f_plus REAL,
            fukui_f_minus REAL,
            fukui_f_zero REAL
        '''
        
        # Combine all node columns
        all_node_columns = base_node_columns
        if npa_nbo_common_columns:
            all_node_columns += "," + npa_nbo_common_columns
        if npa_specific_columns:
            all_node_columns += "," + npa_specific_columns  
        if nbo_specific_columns:
            all_node_columns += "," + nbo_specific_columns
        all_node_columns += "," + catalyst_node_columns
        
        # Add constraints at the end
        all_node_columns += ''',
            PRIMARY KEY (graph_id, atom_index),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS nodes ({all_node_columns})')
        
        # Edges table
        base_edge_columns = '''
            edge_id TEXT,
            graph_id TEXT,
            atom_i INTEGER,
            atom_j INTEGER,
            distance REAL,
            edge_type TEXT
        '''
        
        # graph-specific edge columns
        edge_specific_columns = '''
            bond_order REAL,
            conventional_bond_order REAL,
            bonding_orbital_occupancy REAL,
            bonding_orbital_energy REAL,
            antibonding_orbital_occupancy REAL,
            antibonding_orbital_energy REAL,
            num_2C_BDs INTEGER
        '''
        
        # Combine all edge columns
        all_edge_columns = base_edge_columns
        if edge_specific_columns:
            all_edge_columns += "," + edge_specific_columns
        
        # Add constraints at the end
        all_edge_columns += ''',
            PRIMARY KEY (graph_id, atom_i, atom_j),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS edges ({all_edge_columns})')
        
        # Targets table
        target_columns = '''
            graph_id TEXT,
            frequencies TEXT,  -- JSON array of frequencies
            num_frequencies INTEGER,
            moment_1 REAL,
            moment_2 REAL,
            moment_3 REAL,
            rot_1 REAL,
            rot_2 REAL,
            rot_3 REAL,
            rot_temp_1 REAL,
            rot_temp_2 REAL,
            rot_temp_3 REAL,
            heat_capacity_Cv REAL,
            heat_capacity_Cp REAL,
            entropy REAL,
            ZPE REAL,
            electronic_energy REAL,
            potential_energy REAL,
            potential_energy_correction REAL,
            enthalpy REAL,
            enthalpy_correction REAL,
            gibbs_free_energy REAL,
            gibbs_free_energy_correction REAL,
        '''

        target_columns += '''
            HOMO REAL,
            LUMO REAL,
            HOMO_LUMO_gap REAL,
            buried_volume REAL,
            fraction_vbur REAL,
            free_volume REAL,
            sasa_area REAL,
            sasa_volume REAL,
            IR_freq_mean REAL,
            IR_freq_max REAL,
            IR_freq_min REAL,
            IR_freq_std REAL,
            IR_freq_median REAL,
            IR_eps_mean REAL,
            IR_eps_max REAL,
            IR_eps_min REAL,
            IR_eps_std REAL,
            IR_eps_median REAL,
            IR_intensity_mean REAL,
            IR_intensity_max REAL,
            IR_intensity_min REAL,
            IR_intensity_std REAL,
            IR_intensity_median REAL,
            IR_t2_mean REAL,
            IR_t2_max REAL,
            IR_t2_min REAL,
            IR_t2_std REAL,
            IR_t2_median REAL,
            dipole_x REAL,
            dipole_y REAL,
            dipole_z REAL,
            dipole_magnitude REAL,
            isotropic_polarizability REAL,
            ionization_energy REAL,
            electron_affinity REAL,
            hardness REAL,
            chemical_potential REAL,
            electronegativity REAL,
            electrophilicity REAL,
        '''
        
        target_columns += '''
            natural_minimal_basis REAL,
            natural_rydberg_basis REAL,
            total_core_population REAL,
            total_valence_population REAL,
            total_rydberg_population REAL,
            total_population REAL,
        '''
        
        target_columns += '''
            PRIMARY KEY (graph_id),
            FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
        '''
        
        cursor.execute(f'CREATE TABLE IF NOT EXISTS targets ({target_columns})')
        
        # Labels table - for ML pipeline
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                graph_id TEXT,
                label_name TEXT,
                label_value REAL,
                label_type TEXT,  -- 'regression', 'classification'
                PRIMARY KEY (graph_id, label_name),
                FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
            )
        ''')
        # Create ML views for quantitative data only (excluding categorical features)
        cls._create_ML_views(cursor)

        # Create indices for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_atomic_number ON nodes (atomic_number)')
        
        # Only create bond_order index if bond_order column exists
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bond_order ON edges (bond_order)')
            
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_label_type ON labels (label_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_smiles ON graphs (smiles)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_molecular_mass ON graphs (molecular_mass)')
        
        conn.commit()
        conn.close()
        logger.info(f"\tCreated molecular graph database: {db_path}")
    
    @classmethod
    def _create_ML_views(cls, cursor) -> None:
        """
        Create ML-specific views that filter out non-quantitative features.
        
        Args:
            cursor: SQLite cursor
        """
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS ml_graphs AS
            SELECT 
                graph_id,
                molecular_mass,
                num_atoms,
                num_electrons,
                charge
            FROM graphs
        ''')
        
        base_ml_node_select = '''
            graph_id,
            atom_index,
            atomic_number,
            x_position,
            y_position,
            z_position,
            atomic_mass,
            electronegativity,
            covalent_radius
        '''
        
        ml_node_select = base_ml_node_select
        ml_node_select += '''
            , wiberg_bond_order_total,
            bound_hydrogens,
            node_degree,
            electron_population,
            nmb_population,
            npa_charge,
            natural_charge,
            core_population,
            valence_population,
            rydberg_population,
            total_population,
            core_orbital_occupancy,
            core_orbital_energy,
            lone_pair_1_occupancy,
            lone_pair_1_energy,
            lone_pair_2_occupancy,
            lone_pair_2_energy,
            is_carbene_center,
            proton_isotropic,
            proton_anisotropic,
            carbon_isotropic,
            carbon_anisotropic,
            fukui_f_plus,
            fukui_f_minus,
            fukui_f_zero
        '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_nodes AS
            SELECT {ml_node_select}
            FROM nodes
        ''')
        
        base_ml_edge_select = '''
            graph_id,
            atom_i,
            atom_j,
            distance
        '''
        
        ml_edge_select = base_ml_edge_select
        ml_edge_select += '''
            , bond_order,
            conventional_bond_order,
            bonding_orbital_occupancy,
            bonding_orbital_energy,
            antibonding_orbital_occupancy,
            antibonding_orbital_energy,
            num_2C_BDs
        '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_edges AS
            SELECT {ml_edge_select}
            FROM edges
        ''')
        
        ml_target_select = '''
            graph_id,
            num_frequencies,
            moment_1, moment_2, moment_3,
            rot_1, rot_2, rot_3,
            rot_temp_1, rot_temp_2, rot_temp_3,
            heat_capacity_Cv,
            heat_capacity_Cp,
            entropy,
            ZPE,
            electronic_energy,
            potential_energy,
            potential_energy_correction,
            enthalpy,
            enthalpy_correction,
            gibbs_free_energy,
            gibbs_free_energy_correction
        '''
        
        ml_target_select += '''
            , natural_minimal_basis,
            natural_rydberg_basis,
            total_core_population,
            total_valence_population,
            total_rydberg_population,
            total_population,
            HOMO,
            LUMO,
            HOMO_LUMO_gap,
            buried_volume,
            fraction_vbur,
            free_volume,
            sasa_area,
            sasa_volume,
            IR_freq_mean,
            IR_freq_max,
            IR_freq_min,
            IR_freq_std,
            IR_freq_median,
            IR_eps_mean,
            IR_eps_max,
            IR_eps_min,
            IR_eps_std,
            IR_eps_median,
            IR_intensity_mean,
            IR_intensity_max,
            IR_intensity_min,
            IR_intensity_std,
            IR_intensity_median,
            IR_t2_mean,
            IR_t2_max,
            IR_t2_min,
            IR_t2_std,
            IR_t2_median,
            dipole_x,
            dipole_y,
            dipole_z,
            dipole_magnitude,
            isotropic_polarizability,
            ionization_energy,
            electron_affinity,
            hardness,
            chemical_potential,
            electronegativity,
            electrophilicity
        '''
        
        cursor.execute(f'''
            CREATE VIEW IF NOT EXISTS ml_targets AS
            SELECT {ml_target_select}
            FROM targets
        ''')
        
        logger.info("\tCreated ML views for quantitative features only")
    
    def save_to_database(self, db_path: Union[str, Path]) -> None:
        """
        Save molecular graph to SQLite database using the new structured schema.
        
        Args:
            db_path: Path to SQLite database file
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            graph_features = self.get_graph_features()
            node_features = self.get_node_features()
            edge_features = self.get_edge_features()
            target_features = self.get_target_features()
            
            # Extract SMILES from labels data if available (should be stored in graphs table, not labels)
            smiles_value = graph_features.get('smiles')
            if hasattr(self, '_labels_data') and self._labels_data:
                # Check for SMILES in labels data and use it for the graphs table
                if 'SMILES' in self._labels_data:
                    smiles_value = self._labels_data['SMILES']
                elif 'smiles' in self._labels_data:
                    smiles_value = self._labels_data['smiles']
            
            # Insert graph-level data
            cursor.execute('''
                INSERT OR REPLACE INTO graphs 
                (graph_id, smiles, formula, molecular_mass, num_atoms, 
                 num_electrons, charge, num_bonds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.id,
                smiles_value,
                graph_features.get('formula'),
                graph_features.get('molecular_mass'),
                graph_features.get('num_atoms'),
                graph_features.get('num_electrons'),
                graph_features.get('charge'),
                self.num_bonds
            ))
            
            # Insert node data with all features
            for node in node_features:
                # Validate and clean node data
                atom_index = node.get('atom_index', 0)
                atomic_number = node.get('atomic_number')
                atom_label = node.get('atom_label')
                
                # Ensure proper types
                try:
                    atom_index = int(atom_index) if atom_index is not None else 0
                    atomic_number = int(atomic_number) if atomic_number is not None else None
                    atom_label = str(atom_label) if atom_label is not None else None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid node data for atom {atom_index}: {e}")
                    continue
                
                # Extract coordinates from position field
                position = node.get('position', [None, None, None])
                x_pos = position[0] if position and len(position) > 0 else None
                y_pos = position[1] if position and len(position) > 1 else None  
                z_pos = position[2] if position and len(position) > 2 else None
                
                # Prepare base node data
                node_data = [
                    f"{self.id}_{atom_index}",  # node_id
                    self.id,  # graph_id
                    atom_index,
                    atomic_number,
                    atom_label,
                    x_pos,
                    y_pos, 
                    z_pos,
                    node.get('atomic_mass'),
                    node.get('electronegativity'),
                    node.get('covalent_radius')
                ]
                
                node_data.extend([
                    scalar_wrapper(node.get('wiberg_bond_order_total')),
                    scalar_wrapper(node.get('bound_hydrogens')),
                    scalar_wrapper(node.get('node_degree')),
                    node.get('electron_population'),
                    node.get('nmb_population'),
                    node.get('npa_charge'),
                    node.get('natural_charge'),
                    node.get('core_population'),
                    node.get('valence_population'),
                    node.get('rydberg_population'),
                    node.get('total_population'),
                    node.get('core_orbital_occupancy'),
                    node.get('core_orbital_energy'),
                    node.get('lone_pair_1_occupancy'),
                    node.get('lone_pair_1_energy'),
                    node.get('lone_pair_2_occupancy'),
                    node.get('lone_pair_2_energy'),
                    node.get('is_carbene_center'),
                    json_wrapper(node.get('ACSF_values')),
                    json_wrapper(node.get('SOAP_values')),
                    node.get('proton_isotropic'),
                    node.get('proton_anisotropic'),
                    node.get('carbon_isotropic'),
                    node.get('carbon_anisotropic'),
                    node.get('fukui_f_plus'),
                    node.get('fukui_f_minus'),
                    node.get('fukui_f_zero')
                ])
                
                # Dynamic query based on graph type
                placeholders = ','.join(['?'] * len(node_data))
                try:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO nodes VALUES ({placeholders})
                    ''', node_data)
                except Exception as e:
                    logger.error(f"Node insertion error for {self.id}: {e}")
                    logger.error(f"Node data: {node_data}")
                    logger.error(f"Node data types: {[type(x) for x in node_data]}")
                    raise
        
            for edge_idx, edge in enumerate(edge_features):
                try:
                    # Validate and clean edge data
                    atom_i = edge.get('atom_i')
                    atom_j = edge.get('atom_j')
                    
                    if isinstance(atom_i, (tuple, list)):
                        atom_i = atom_i[0] if len(atom_i) > 0 else 0
                    if isinstance(atom_j, (tuple, list)):
                        atom_j = atom_j[0] if len(atom_j) > 0 else 0
                    
                    try:
                        atom_i = int(atom_i) if atom_i is not None else 0
                        atom_j = int(atom_j) if atom_j is not None else 0
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid atom indices for edge: atom_i={atom_i}, atom_j={atom_j}")
                        continue
                    
                    distance = edge.get('distance')
                    
                    if hasattr(distance, 'item') and callable(distance.item):
                        distance = distance.item()
                    elif isinstance(distance, (np.floating, np.integer)):
                        distance = float(distance)
                    
                    edge_type = edge.get('edge_type')
                    
                    # Prepare base edge data
                    edge_data = [
                        f"{self.id}_{atom_i}_{atom_j}",  # edge_id
                        self.id,  # graph_id
                        atom_i,
                        atom_j,
                        distance,
                        edge_type
                    ]
                except Exception as edge_e:
                    logger.error(f"Failed to process edge {edge_idx} for {self.id}: {edge_e}")
                    logger.error(f"Edge data: {repr(edge)}")
                    # Continue with next edge instead of failing entire molecule
                    continue

                features = [
                    edge.get('bond_order'),
                    edge.get('conventional_bond_order'),
                    edge.get('bonding_orbital_occupancy'),
                    edge.get('bonding_orbital_energy'),
                    edge.get('antibonding_orbital_occupancy'),
                    edge.get('antibonding_orbital_energy')
                ]
                edge_data.extend(features)

                num_2c_bds = edge.get('num_2C_BDs')
                if isinstance(num_2c_bds, (tuple, list)):
                    num_2c_bds = num_2c_bds[0] if len(num_2c_bds) > 0 else None
                edge_data.append(num_2c_bds)
                
                # Dynamic query based on features
                placeholders = ','.join(['?'] * len(edge_data))
                
                # Insert edge into database
                try:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO edges VALUES ({placeholders})
                    ''', edge_data)
                except Exception as e:
                    logger.error(f"Edge insertion error for {self.id} edge {atom_i}-{atom_j}: {e}")
                    logger.error(f"Edge data: {edge_data}")
                    logger.error(f"Edge data types: {[type(x) for x in edge_data]}")
                    continue
            
            # Insert target data
            target_data = [
                self.id,
                json.dumps(target_features.get('frequencies', [])),
                target_features.get('num_frequencies'),
                target_features.get('moment_1'),
                target_features.get('moment_2'),
                target_features.get('moment_3'),
                target_features.get('rot_1'),
                target_features.get('rot_2'),
                target_features.get('rot_3'),
                target_features.get('rot_temp_1'),
                target_features.get('rot_temp_2'),
                target_features.get('rot_temp_3'),
                target_features.get('heat_capacity_Cv'),
                target_features.get('heat_capacity_Cp'),
                target_features.get('entropy'),
                target_features.get('ZPE'),
                target_features.get('electronic_energy'),
                target_features.get('potential_energy'),
                target_features.get('potential_energy_correction'),
                target_features.get('enthalpy'),
                target_features.get('enthalpy_correction'),
                target_features.get('gibbs_free_energy'),
                target_features.get('gibbs_free_energy_correction')
            ]
            
            target_data.extend([
                target_features.get('HOMO'),
                target_features.get('LUMO'),
                target_features.get('HOMO_LUMO_gap'),
                target_features.get('buried_volume'),
                target_features.get('fraction_vbur'),
                target_features.get('free_volume'),
                target_features.get('sasa_area'),
                target_features.get('sasa_volume'),
                target_features.get('IR_freq_mean'),
                target_features.get('IR_freq_max'),
                target_features.get('IR_freq_min'),
                target_features.get('IR_freq_std'),
                target_features.get('IR_freq_median'),
                target_features.get('IR_eps_mean'),
                target_features.get('IR_eps_max'),
                target_features.get('IR_eps_min'),
                target_features.get('IR_eps_std'),
                target_features.get('IR_eps_median'),
                target_features.get('IR_intensity_mean'),
                target_features.get('IR_intensity_max'),
                target_features.get('IR_intensity_min'),
                target_features.get('IR_intensity_std'),
                target_features.get('IR_intensity_median'),
                target_features.get('IR_t2_mean'),
                target_features.get('IR_t2_max'),
                target_features.get('IR_t2_min'),
                target_features.get('IR_t2_std'),
                target_features.get('IR_t2_median'),
                target_features.get('dipole_x'),
                target_features.get('dipole_y'),
                target_features.get('dipole_z'),
                target_features.get('dipole_magnitude'),
                target_features.get('isotropic_polarizability'),
                target_features.get('ionization_energy'),
                target_features.get('electron_affinity'),
                target_features.get('hardness'),
                target_features.get('chemical_potential'),
                target_features.get('electronegativity'),
                target_features.get('electrophilicity'),
                target_features.get('natural_minimal_basis'),
                target_features.get('natural_rydberg_basis'),
                target_features.get('total_core_population'),
                target_features.get('total_valence_population'),
                target_features.get('total_rydberg_population'),
                target_features.get('total_population')
            ])
            
            placeholders = ','.join(['?'] * len(target_data))
            cursor.execute(f'''
                INSERT OR REPLACE INTO targets VALUES ({placeholders})
            ''', target_data)
            
            # Insert labels data if available
            if hasattr(self, '_labels_data') and self._labels_data:
                logger.debug(f"Inserting labels for {self.id}: {self._labels_data}")
                for label_name, label_value in self._labels_data.items():
                    # Skip mol_id
                    if label_name == 'mol_id':
                        continue
                    
                    if label_name.lower() in ['smiles']:
                        continue
                    
                    # Determine label type based on value
                    numeric_value = None
                    if isinstance(label_value, (int, float)):
                        numeric_value = float(label_value)
                        label_type = 'regression'
                    else:
                        # Try to convert string to numeric
                        try:
                            numeric_value = float(label_value)
                            label_type = 'regression'
                        except (ValueError, TypeError):
                            # Not numeric, treat as classification
                            label_type = 'classification'
                            # For now, skip non-numeric labels
                            logger.debug(f"Skipping non-numeric label {label_name}={label_value}")
                            continue
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO labels 
                        (graph_id, label_name, label_value, label_type)
                        VALUES (?, ?, ?, ?)
                    ''', (self.id, label_name, numeric_value, label_type))
            else:
                logger.debug(f"No labels data for {self.id}: hasattr={hasattr(self, '_labels_data')}, data={getattr(self, '_labels_data', None)}")
            
            conn.commit()
            logger.info(f"\tSaved graph {self.id} to database")
            
        except Exception as e:
            logger.error(f"Database transaction failed for {self.id}: {e}")
            logger.exception("Full traceback:")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @classmethod
    def load_from_database(cls, graph_id: str, db_path: Union[str, Path]) -> 'MolecularGraph':
        """
        Load molecular graph from SQLite database.
        
        Args:
            graph_id: Graph identifier to load
            db_path: Path to SQLite database file
            
        Returns:
            MolecularGraph: Loaded graph object
        """
        conn = sqlite3.connect(db_path)
        try:
            # Load graph metadata
            graph_query = "SELECT * FROM graphs WHERE graph_id = ?"
            graph_data = pd.read_sql_query(graph_query, conn, params=[graph_id])
            
            if graph_data.empty:
                raise ValueError(f"Graph {graph_id} not found in database")
            
            graph_row = graph_data.iloc[0]
            
            # Load nodes
            nodes_query = "SELECT * FROM nodes WHERE graph_id = ? ORDER BY atom_index"
            nodes_data = pd.read_sql_query(nodes_query, conn, params=[graph_id])
            
            # Load edges  
            edges_query = "SELECT * FROM edges WHERE graph_id = ? ORDER BY atom_i, atom_j"
            edges_data = pd.read_sql_query(edges_query, conn, params=[graph_id])
            
            # Load targets
            targets_query = "SELECT * FROM targets WHERE graph_id = ?"
            targets_data = pd.read_sql_query(targets_query, conn, params=[graph_id])
        finally:
            conn.close()

        def _maybe_json(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        node_rows = nodes_data.to_dict(orient="records") if not nodes_data.empty else []
        edge_rows = edges_data.to_dict(orient="records") if not edges_data.empty else []
        target_row = targets_data.iloc[0].to_dict() if not targets_data.empty else {}

        def _column_values(rows, column_name):
            return [row.get(column_name) for row in rows] if rows else None

        qm_data = {
            "id": graph_id,
            "smiles": graph_row.get("smiles"),
            "formula": graph_row.get("formula"),
            "molecular_mass": graph_row.get("molecular_mass"),
            "num_atoms": int(graph_row["num_atoms"]) if pd.notna(graph_row.get("num_atoms")) else len(node_rows),
            "atomic_numbers": [row.get("atomic_number") for row in node_rows] if node_rows else None,
            "atom_labels": [row.get("atom_label") for row in node_rows] if node_rows else None,
            "xyz_coordinates": [
                [row.get("x_position"), row.get("y_position"), row.get("z_position")]
                for row in node_rows
            ] if node_rows else None,
            "wiberg_bond_order_totals": _column_values(node_rows, "wiberg_bond_order_total"),
            "bound_hydrogens": _column_values(node_rows, "bound_hydrogens"),
            "node_degrees": _column_values(node_rows, "node_degree"),
            "electron_populations": _column_values(node_rows, "electron_population"),
            "nmb_populations": _column_values(node_rows, "nmb_population"),
            "npa_charges": _column_values(node_rows, "npa_charge"),
            "natural_charges": _column_values(node_rows, "natural_charge"),
            "core_populations": _column_values(node_rows, "core_population"),
            "valence_populations": _column_values(node_rows, "valence_population"),
            "rydberg_populations": _column_values(node_rows, "rydberg_population"),
            "total_populations": _column_values(node_rows, "total_population"),
            "core_orbital_occupancies": _column_values(node_rows, "core_orbital_occupancy"),
            "core_orbital_energies": _column_values(node_rows, "core_orbital_energy"),
            "lone_pair_occupancies": [[row.get("lone_pair_1_occupancy"), row.get("lone_pair_2_occupancy")] for row in node_rows] if node_rows else None,
            "lone_pair_energies": [[row.get("lone_pair_1_energy"), row.get("lone_pair_2_energy")] for row in node_rows] if node_rows else None,
            "polarizability": {"isotropic_polarizability": target_row.get("isotropic_polarizability")},
            "homo_lumo": {
                "HOMO": target_row.get("HOMO"),
                "LUMO": target_row.get("LUMO"),
                "HOMO_LUMO_gap": target_row.get("HOMO_LUMO_gap"),
            },
            "moments_of_inertia": [target_row.get("moment_1"), target_row.get("moment_2"), target_row.get("moment_3")],
            "rotational_constants": [target_row.get("rot_1"), target_row.get("rot_2"), target_row.get("rot_3")],
            "rotational_temperatures": [target_row.get("rot_temp_1"), target_row.get("rot_temp_2"), target_row.get("rot_temp_3")],
            "morfeus": {
                "buried_volume": target_row.get("buried_volume"),
                "fraction_vbur": target_row.get("fraction_vbur"),
                "free_volume": target_row.get("free_volume"),
                "sasa_area": target_row.get("sasa_area"),
                "sasa_volume": target_row.get("sasa_volume"),
            },
            "IR_stats": {
                "frequencies": {
                    "mean": target_row.get("IR_freq_mean"),
                    "max": target_row.get("IR_freq_max"),
                    "min": target_row.get("IR_freq_min"),
                    "std": target_row.get("IR_freq_std"),
                    "median": target_row.get("IR_freq_median"),
                },
                "eps": {
                    "mean": target_row.get("IR_eps_mean"),
                    "max": target_row.get("IR_eps_max"),
                    "min": target_row.get("IR_eps_min"),
                    "std": target_row.get("IR_eps_std"),
                    "median": target_row.get("IR_eps_median"),
                },
                "intensity": {
                    "mean": target_row.get("IR_intensity_mean"),
                    "max": target_row.get("IR_intensity_max"),
                    "min": target_row.get("IR_intensity_min"),
                    "std": target_row.get("IR_intensity_std"),
                    "median": target_row.get("IR_intensity_median"),
                },
                "t2": {
                    "mean": target_row.get("IR_t2_mean"),
                    "max": target_row.get("IR_t2_max"),
                    "min": target_row.get("IR_t2_min"),
                    "std": target_row.get("IR_t2_std"),
                    "median": target_row.get("IR_t2_median"),
                },
            },
            "dipole_moment": {
                "dipole_x": target_row.get("dipole_x"),
                "dipole_y": target_row.get("dipole_y"),
                "dipole_z": target_row.get("dipole_z"),
                "dipole_magnitude": target_row.get("dipole_magnitude"),
            },
            "IE_EA": {
                "ionization_energy": target_row.get("ionization_energy"),
                "electron_affinity": target_row.get("electron_affinity"),
                "hardness": target_row.get("hardness"),
                "chemical_potential": target_row.get("chemical_potential"),
                "electronegativity": target_row.get("electronegativity"),
                "electrophilicity": target_row.get("electrophilicity"),
            },
            "ACSF": [_maybe_json(row.get("ACSF_values")) for row in node_rows] if node_rows else None,
            "SOAP": [_maybe_json(row.get("SOAP_values")) for row in node_rows] if node_rows else None,
            "Fukui_indices": {
                "f_plus": [row.get("fukui_f_plus") for row in node_rows] if node_rows else None,
                "f_minus": [row.get("fukui_f_minus") for row in node_rows] if node_rows else None,
                "f_zero": [row.get("fukui_f_zero") for row in node_rows] if node_rows else None,
            },
            "frequencies": _maybe_json(target_row.get("frequencies")),
            "num_frequencies": target_row.get("num_frequencies"),
            "moment_1": target_row.get("moment_1"),
            "moment_2": target_row.get("moment_2"),
            "moment_3": target_row.get("moment_3"),
            "rot_1": target_row.get("rot_1"),
            "rot_2": target_row.get("rot_2"),
            "rot_3": target_row.get("rot_3"),
            "rot_temp_1": target_row.get("rot_temp_1"),
            "rot_temp_2": target_row.get("rot_temp_2"),
            "rot_temp_3": target_row.get("rot_temp_3"),
            "heat_capacity_Cv": target_row.get("heat_capacity_Cv"),
            "heat_capacity_Cp": target_row.get("heat_capacity_Cp"),
            "entropy": target_row.get("entropy"),
            "ZPE": target_row.get("ZPE"),
            "electronic_energy": target_row.get("electronic_energy"),
            "potential_energy": target_row.get("potential_energy"),
            "potential_energy_correction": target_row.get("potential_energy_correction"),
            "enthalpy": target_row.get("enthalpy"),
            "enthalpy_correction": target_row.get("enthalpy_correction"),
            "gibbs_free_energy": target_row.get("gibbs_free_energy"),
            "gibbs_free_energy_correction": target_row.get("gibbs_free_energy_correction"),
            "natural_minimal_basis": target_row.get("natural_minimal_basis"),
            "natural_rydberg_basis": target_row.get("natural_rydberg_basis"),
            "total_core_population": target_row.get("total_core_population"),
            "total_valence_population": target_row.get("total_valence_population"),
            "total_rydberg_population": target_row.get("total_rydberg_population"),
            "total_population": target_row.get("total_population"),
        }

        dict_data = DictData(qm_data)
        molecular_graph = cls(dict_data)
        molecular_graph._labels_data = {}

        return molecular_graph
    
    def load_labels(self, db_path: Union[str, Path], label_names: Optional[List[str]] = None) -> Dict:
        """
        Load labels for this graph from database.
        
        Args:
            db_path: Path to SQLite database file
            label_names: Specific label names to load, or None for all
            
        Returns:
            Dict: Label name to value mapping
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            if label_names:
                placeholders = ','.join('?' * len(label_names))
                cursor.execute(f'''
                    SELECT label_name, label_value FROM labels 
                    WHERE graph_id = ? AND label_name IN ({placeholders})
                ''', [self.id] + label_names)
            else:
                cursor.execute('SELECT label_name, label_value FROM labels WHERE graph_id = ?', (self.id,))
            
            return dict(cursor.fetchall())
        finally:
            conn.close()
    
    @classmethod
    def query_database(cls, db_path: Union[str, Path], 
                      conditions: Optional[Dict] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Query molecular graphs from database with conditions.
        
        Args:
            db_path: Path to SQLite database file
            conditions: Dict of column:value conditions to filter by
            limit: Maximum number of results to return
            
        Returns:
            pd.DataFrame: Query results
        """
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM graphs"
        params = []
        
        if conditions:
            where_clauses = []
            for col, val in conditions.items():
                where_clauses.append(f"{col} = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    @classmethod
    def get_ml_data(cls, db_path: Union[str, Path], 
                   graph_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get ML-ready data (quantitative features only) from database views.
        
        Args:
            db_path: Path to SQLite database file
            graph_ids: Filter by specific graph IDs
            
        Returns:
            Dict containing DataFrames for graphs, nodes, edges, and targets
        """
        conn = sqlite3.connect(db_path)
        
        try:
            where_conditions = []
            params = []
                        
            if graph_ids:
                placeholders = ','.join('?' * len(graph_ids))
                where_conditions.append(f"graph_id IN ({placeholders})")
                params.extend(graph_ids)
            
            where_clause = ""
            if where_conditions:
                where_clause = " WHERE " + " AND ".join(where_conditions)
            
            # Query ML views
            ml_data = {}
            
            # Get graph-level ML features
            ml_data['graphs'] = pd.read_sql_query(
                f"SELECT * FROM ml_graphs{where_clause}", 
                conn, params=params
            )
            
            # Get node-level ML features
            if graph_ids:
                ml_data['nodes'] = pd.read_sql_query(
                    f"SELECT * FROM ml_nodes{where_clause}", 
                    conn, params=params
                )
            else:
                ml_data['nodes'] = pd.read_sql_query("SELECT * FROM ml_nodes", conn)
            
            # Get edge-level ML features  
            if graph_ids:
                ml_data['edges'] = pd.read_sql_query(
                    f"SELECT * FROM ml_edges{where_clause}", 
                    conn, params=params
                )
            else:
                ml_data['edges'] = pd.read_sql_query("SELECT * FROM ml_edges", conn)
            
            # Get target ML features
            ml_data['targets'] = pd.read_sql_query(
                f"SELECT * FROM ml_targets{where_clause}", 
                conn, params=params
            )
            
            return ml_data
            
        finally:
            conn.close()
    
    @classmethod 
    def get_feature_summary(cls, db_path: Union[str, Path]) -> Dict:
        """
        Get summary statistics of features in the database.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            Dict: Feature summary statistics
        """
        conn = sqlite3.connect(db_path)
        
        try:
            summary = {}
            
            # Molecular mass distribution
            mass_stats = pd.read_sql_query(
                "SELECT MIN(molecular_mass) as min_mass, MAX(molecular_mass) as max_mass, AVG(molecular_mass) as avg_mass FROM graphs", 
                conn
            ).iloc[0]
            summary['molecular_mass'] = mass_stats.to_dict()
            
            # Atom count distribution
            atom_stats = pd.read_sql_query(
                "SELECT MIN(num_atoms) as min_atoms, MAX(num_atoms) as max_atoms, AVG(num_atoms) as avg_atoms FROM graphs", 
                conn
            ).iloc[0]
            summary['atom_counts'] = atom_stats.to_dict()
            
            # Atomic number distribution
            element_dist = pd.read_sql_query(
                "SELECT atomic_number, COUNT(*) as count FROM nodes GROUP BY atomic_number ORDER BY count DESC", 
                conn
            )
            summary['element_distribution'] = dict(zip(element_dist['atomic_number'], element_dist['count']))
            
            # Bond order distribution (if available)
            try:
                bond_stats = pd.read_sql_query(
                    "SELECT MIN(bond_order) as min_bo, MAX(bond_order) as max_bo, AVG(bond_order) as avg_bo FROM edges WHERE bond_order IS NOT NULL", 
                    conn
                ).iloc[0]
                summary['bond_orders'] = bond_stats.to_dict()
            except:
                summary['bond_orders'] = None
                
            return summary
            
        finally:
            conn.close()
    
    @classmethod
    def export_ml_features(cls, db_path: Union[str, Path], 
                          output_dir: Union[str, Path],
                          ) -> None:
        """
        Export ML-ready features to CSV files.
        
        Args:
            db_path: Path to SQLite database file
            output_dir: Directory to save CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ml_data = cls.get_ml_data(db_path)
        
        # Export each feature level
        for level, df in ml_data.items():
            if not df.empty:
                filename = f"ml_{level}"
                filename += ".csv"
                
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"\tExported {len(df)} {level} features to {filepath}")
        
        # Export combined dataset for easy loading
        if all(not df.empty for df in ml_data.values()):
            combined_filename = "ml_combined"
            combined_filename += ".pkl"
            
            combined_path = output_dir / combined_filename
            with open(combined_path, 'wb') as f:
                import pickle
                pickle.dump(ml_data, f)
            logger.info(f"\tExported combined ML dataset to {combined_path}")
    
    def to_pytorch_geometric(self, labels_df: Optional[pd.DataFrame] = None) -> Optional['Data']:
        """
        Convert to PyTorch Geometric Data object for GNN training.
        
        Args:
            labels_df: DataFrame with graph_id and target columns
            
        Returns:
            Data: PyTorch Geometric data object or None if torch unavailable
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, cannot create Data object")
            return None
        
        ml_data = self.get_ml_features()
        
        # Node features tensor
        node_features = []
        for node in ml_data['node_features']:
            features = [float(v) if not np.isnan(v) else 0.0 for v in node.values()]
            node_features.append(features)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge connectivity and features
        edge_indices = []
        edge_features = []
        for edge in ml_data['edge_features']:
            edge_copy = edge.copy()  # Don't modify original
            i, j = int(edge_copy.pop('atom_i', 0)), int(edge_copy.pop('atom_j', 0))
            features = [float(v) if not np.isnan(v) else 0.0 for v in edge_copy.values()]
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([features, features])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T if edge_indices else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.zeros((0, 1))
        
        # Graph-level features
        graph_features = [float(v) if not np.isnan(v) else 0.0 for v in ml_data['graph_features'].values()]
        graph_attr = torch.tensor(graph_features, dtype=torch.float) if graph_features else None
        
        # Target labels if provided
        y = None
        if labels_df is not None and self.id in labels_df['graph_id'].values:
            target_row = labels_df[labels_df['graph_id'] == self.id]
            if not target_row.empty:
                y = torch.tensor([float(target_row.iloc[0]['target'])], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_attr=graph_attr)
    
    def to_networkx(self) -> Optional['nx.Graph']:
        """
        Convert to NetworkX graph for analysis.
        
        Returns:
            nx.Graph: NetworkX graph object or None if networkx unavailable
        """
        if not HAS_VISUALIZATION:
            logger.warning("NetworkX not available")
            return None
        
        G = nx.Graph()
        
        # Add nodes with features
        for node in self.get_node_features():
            node_id = node.get('atom_index', 0)
            G.add_node(node_id, **node)
        
        # Add edges with features  
        for edge in self.get_edge_features():
            i, j = edge.get('atom_i', 0), edge.get('atom_j', 0)
            G.add_edge(i, j, **edge)
        
        # Add graph-level attributes
        G.graph.update(self.get_graph_features())
        
        return G
        
    def summary(self) -> Dict:
        """
        Get molecular graph summary statistics.
        
        Returns:
            Dict: Summary statistics
        """
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()
        
        summary = {
            'graph_id': self.id,
            'num_atoms': len(node_features),
            'num_bonds': len(edge_features),
            'formula': self.get_graph_features().get('formula'),
            'molecular_mass': self.get_graph_features().get('molecular_mass')
        }
        
        if node_features:
            atom_types = {}
            for node in node_features:
                atom_label = node.get('atom_label', 'Unknown')
                atom_types[atom_label] = atom_types.get(atom_label, 0) + 1
            summary['atom_composition'] = atom_types
        
        if edge_features:
            bond_orders = [edge.get('wiberg_bond_order', 0) for edge in edge_features if edge.get('wiberg_bond_order') is not None]
            if bond_orders:
                summary['avg_bond_order'] = np.mean(bond_orders)
                summary['max_bond_order'] = np.max(bond_orders)
        
        return summary
    
    @classmethod
    def delete_graph(cls, graph_id: str, db_path: Union[str, Path]) -> bool:
        """
        Delete a specific molecular graph and all its associated data from the database.
        
        Args:
            graph_id: ID of the graph to delete
            db_path: Path to SQLite database file
            
        Returns:
            bool: True if graph was deleted, False if graph was not found
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Check if graph exists
            cursor.execute("SELECT COUNT(*) FROM graphs WHERE graph_id = ?", (graph_id,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.warning(f"Graph '{graph_id}' not found in database")
                return False
            
            # Delete from all tables (foreign key constraints will handle cascading)
            # Delete edges first
            cursor.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
            edges_deleted = cursor.rowcount
            
            # Delete nodes
            cursor.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
            nodes_deleted = cursor.rowcount
            
            # Delete target features
            cursor.execute("DELETE FROM targets WHERE graph_id = ?", (graph_id,))
            targets_deleted = cursor.rowcount
            
            # Delete graph
            cursor.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
            graphs_deleted = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Successfully deleted graph '{graph_id}': "
                       f"{graphs_deleted} graph, {nodes_deleted} nodes, "
                       f"{edges_deleted} edges, {targets_deleted} targets")
            
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting graph '{graph_id}': {e}")
            raise
        finally:
            conn.close()
    
    @classmethod
    def delete_graphs_by_criteria(cls, db_path: Union[str, Path], 
                                 where_clause: str = None,
                                 parameters: tuple = None) -> int:
        """
        Delete graphs matching specific criteria.
        
        Args:
            db_path: Path to SQLite database file
            where_clause: SQL WHERE clause (without 'WHERE' keyword)
            parameters: Parameters for the WHERE clause
            
        Returns:
            int: Number of graphs deleted
            
        Examples:
            # Delete graphs with energy > -100
            delete_graphs_by_criteria(db_path, "json_extract(graph_features, '$.total_energy') > ?", (-100,))
            
            # Delete graphs of specific molecule type
            delete_graphs_by_criteria(db_path, "json_extract(graph_features, '$.molecule_type') = ?", ('alkane',))
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        if not where_clause:
            raise ValueError("where_clause is required for safety. Use clear_database() to delete all.")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # First get the graph IDs that match the criteria
            query = f"SELECT graph_id FROM graphs WHERE {where_clause}"
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            graph_ids = [row[0] for row in cursor.fetchall()]
            
            if not graph_ids:
                logger.info("No graphs match the specified criteria")
                return 0
            
            # Delete each graph
            deleted_count = 0
            for graph_id in graph_ids:
                # Delete edges
                cursor.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
                
                # Delete nodes
                cursor.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
                
                # Delete target features
                cursor.execute("DELETE FROM targets WHERE graph_id = ?", (graph_id,))
                
                # Delete graph
                cursor.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
                deleted_count += 1
            
            conn.commit()
            
            logger.info(f"Successfully deleted {deleted_count} graphs matching criteria")
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting graphs by criteria: {e}")
            raise
        finally:
            conn.close()
    
    @classmethod
    def clear_database(cls, db_path: Union[str, Path], confirm: bool = False) -> int:
        """
        Remove all data from the database while keeping the schema.
        
        Args:
            db_path: Path to SQLite database file
            confirm: Must be True to actually clear the database (safety check)
            
        Returns:
            int: Total number of records deleted
        """
        if not confirm:
            raise ValueError("confirm=True is required to clear the database")
        
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Count total records before deletion
            cursor.execute("SELECT COUNT(*) FROM graphs")
            graph_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM targets")
            target_count = cursor.fetchone()[0]
            
            total_records = graph_count + node_count + edge_count + target_count
            
            if total_records == 0:
                logger.info("Database is already empty")
                return 0
            
            # Delete all data
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM nodes") 
            cursor.execute("DELETE FROM targets")
            cursor.execute("DELETE FROM graphs")
            
            conn.commit()
            
            logger.info(f"Successfully cleared database: {graph_count} graphs, "
                       f"{node_count} nodes, {edge_count} edges, {target_count} targets")
            
            return total_records
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            conn.close()
    
    def __repr__(self) -> str:
        """String representation of molecular graph."""
        return f"MolecularGraph(id='{self.id}', atoms={self.num_atoms}, bonds={self.num_bonds})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        summary = self.summary()
        composition = summary.get('atom_composition', {})
        comp_str = ', '.join([f"{count}{atom}" for atom, count in composition.items()])
        return f"MolecularGraph {self.id}: {comp_str} ({self.num_atoms} atoms, {self.num_bonds} bonds)"


class GraphDatabase:
    """Utility class for batch operations on molecular graph databases."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize database interface.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            MolecularGraph.create_database(self.db_path)
    
    def add_labels_from_csv(self, csv_path: Union[str, Path], 
                           id_column: str = 'graph_id',
                           target_column: str = 'target',
                           label_name: str = 'target',
                           label_type: str = 'regression') -> None:
        """
        Add labels to database from CSV file.
        
        Args:
            csv_path: Path to CSV file with labels
            id_column: Column name containing graph IDs
            target_column: Column name containing target values  
            label_name: Name for this label type
            label_type: 'regression' or 'classification'
        """
        # Use polars if available, otherwise pandas
        if HAS_POLARS:
            df = pl.scan_csv(csv_path).select([id_column, target_column]).collect()
            rows = df.rows()
        else:
            df = pd.read_csv(csv_path)[[id_column, target_column]]
            rows = df.itertuples(index=False, name=None)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            count = 0
            for row in rows:
                graph_id, target_value = row
                cursor.execute('''
                    INSERT OR REPLACE INTO labels (graph_id, label_name, label_value, label_type)
                    VALUES (?, ?, ?, ?)
                ''', (graph_id, label_name, float(target_value), label_type))
                count += 1
            
            conn.commit()
            logger.info(f"\tAdded {count} labels from {csv_path}")
        finally:
            conn.close()
    
    def get_ml_dataset(self, label_name: str = 'target') -> Tuple[List['MolecularGraph'], List[float]]:
        """
        Get molecular graphs and labels for ML training.
        
        Args:
            label_name: Name of label to retrieve
            
        Returns:
            Tuple of (graphs, labels) lists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Query for graphs with labels
            query = '''
                SELECT g.graph_id, l.label_value 
                FROM graphs g 
                JOIN labels l ON g.graph_id = l.graph_id 
                WHERE l.label_name = ?
            '''
            params = [label_name]
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            graphs = []
            labels = []
            
            for graph_id, label_value in results:
                try:
                    graph = MolecularGraph.load_from_database(graph_id, self.db_path)
                    graphs.append(graph)
                    labels.append(float(label_value))
                except Exception as e:
                    logger.warning(f"Failed to load graph {graph_id}: {e}")
            
            return graphs, labels
            
        finally:
            conn.close()
    
    @classmethod
    def export_features_csv(cls, db_path: Union[str, Path], output_path: Union[str, Path],
                           feature_level: str = 'graph') -> None:
        """
        Export features to CSV for external ML tools.
        
        Args:
            db_path: Path to SQLite database file
            output_path: Path for output CSV file
            feature_level: 'graph', 'node', or 'edge'
        """
        conn = sqlite3.connect(db_path)
        
        if feature_level == 'graph':
            query = "SELECT graph_id, graph_features FROM graphs"
        elif feature_level == 'node':
            query = "SELECT graph_id, atom_index, node_features FROM nodes"
        elif feature_level == 'edge':
            query = "SELECT graph_id, atom_i, atom_j, edge_features FROM edges"
        else:
            raise ValueError("feature_level must be 'graph', 'node', or 'edge'")
        
        df = pd.read_sql_query(query, conn)
        
        # Expand JSON features into columns
        if not df.empty:
            feature_col = f"{feature_level}_features"
            features_expanded = pd.json_normalize(df[feature_col].apply(json.loads))
            df = pd.concat([df.drop(columns=[feature_col]), features_expanded], axis=1)
        
        df.to_csv(output_path, index=False)
        logger.info(f"\txported {len(df)} {feature_level} features to {output_path}")
        
        conn.close()

    @classmethod
    def map_db_columns_to_code(cls, db_path: Union[str, Path], source_dir: Union[str, Path]):
        """
        Inspect the SQLite schema and scan the source tree for occurrences of each
        column name so we can map DB columns to variables/fields used in code.

        Args:
            db_path: path to the SQLite DB file
            source_dir: path to the project `src/` directory to scan

        Returns:
            Dict: {table_name: {column_name: [list of file:line occurrences]}}
        """
        db_path = Path(db_path)
        source_dir = Path(source_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        mapping = {}
        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            for table in tables:
                mapping[table] = {}
                cursor.execute(f"PRAGMA table_info({table})")
                cols = cursor.fetchall()  # cid, name, type, notnull, dflt_value, pk
                col_names = [c[1] for c in cols]

                # For each column name, search the source tree for usages
                for col in col_names:
                    occurrences = []
                    # Walk source dir and grep simple text matches
                    for p in source_dir.rglob('*.py'):
                        try:
                            with p.open('r', encoding='utf-8') as fh:
                                for i, line in enumerate(fh, start=1):
                                    if col in line:
                                        occurrences.append(f"{p.relative_to(Path.cwd())}:{i}: {line.strip()}")
                        except Exception:
                            continue
                    mapping[table][col] = occurrences
        finally:
            conn.close()

        return mapping

# backward compatibility
Graph = MolecularGraph
