#!/bin/bash

streamlit run revenue_forecaster.py \
    --server.port=$PORT \
    --server.address=0.0.0.0
