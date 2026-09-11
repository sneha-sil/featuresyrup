API Documentation
=================

This section provides detailed API reference for all featuresyrup modules.

Core Classes
------------

.. automodule:: featuresyrup.classes
   :members:
   :undoc-members:
   :show-inheritance:

Data Dictionary Class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: featuresyrup.classes.DictData
   :members:
   :undoc-members:
   :show-inheritance:

Molecular Graph
---------------

.. automodule:: featuresyrup.graph
   :members:
   :undoc-members:
   :show-inheritance:

MolecularGraph Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: featuresyrup.graph.MolecularGraph
   :members:
   :undoc-members:
   :show-inheritance:

Processing Functions
--------------------

.. automodule:: featuresyrup.functions
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: featuresyrup.functions.generate_qm_data_dict

.. autofunction:: featuresyrup.functions.parse_shermo_output

.. autofunction:: featuresyrup.functions.parse_janpa_output

.. autofunction:: featuresyrup.functions.parse_nbo_output

Command Line Interface
----------------------

.. automodule:: featuresyrup.cli
   :members:
   :undoc-members:
   :show-inheritance:

CLI Commands
^^^^^^^^^^^^

.. autoclass:: featuresyrup.cli.FeaturesyrupCLI
   :members:
   :undoc-members:
   :show-inheritance:

Database Operations
-------------------

Database Schema
^^^^^^^^^^^^^^^

featuresyrup creates the following SQLite tables:

**graphs table:**
  - mol_id: Unique molecule identifier
   - graph_type: Catalyst mode label used by the package (``catalyst``)
  - smiles: SMILES string representation
  - formula: Molecular formula
  - num_atoms: Number of atoms
  - num_edges: Number of bonds
  - created_at: Timestamp

**nodes table:**
  - graph_id: Foreign key to graphs table
  - node_id: Node identifier within graph
  - atomic_number: Atomic number
  - element: Element symbol
  - x, y, z: Atomic coordinates
  - charge: Atomic charge (if available)
  - population: Natural population (if available)

**edges table:**
  - graph_id: Foreign key to graphs table
  - edge_id: Edge identifier within graph
  - atom_i, atom_j: Connected atom indices
  - bond_order: Wiberg bond order
  - distance: Interatomic distance

**targets table:**
  - graph_id: Foreign key to graphs table
  - property_name: Name of target property
  - property_value: Numerical value
  - property_units: Units (if applicable)


.. autosummary::
   :toctree: autosummary
   :recursive:

   featuresyrup.cli
   featuresyrup.classes
   featuresyrup.graph
   featuresyrup.functions
