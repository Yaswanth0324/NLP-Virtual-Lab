#!/bin/bash
gunicorn app_single:app --bind 0.0.0.0:5000 --workers 4
