Welcome to Toxic Comment Classification's documentation!
====================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   model_card
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project Overview
---------------

Toxic Comment Classification is a robust machine learning model for detecting toxic comments using RoBERTa, with built-in security features against text obfuscation attacks.

Key Features
-----------

* Advanced Classification using RoBERTa
* Multi-label Support
* Security Features:
  * Text Obfuscation Detection
  * Backtranslation Protection
  * Synonym Attack Resilience
* Comprehensive Evaluation Metrics

Installation
-----------

.. code-block:: bash

   git clone https://github.com/sebtosca/toxic-comment-classification.git
   cd toxic-comment-classification
   pip install -r requirements.txt

Quick Start
----------

.. code-block:: python

   from models.ML.main import main
   main()

For more detailed usage, see the :doc:`api/modules` documentation. 