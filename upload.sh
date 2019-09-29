#!/bin/bash
rsync -rvzP ./dist/ lukas@lw1.at:/var/www/nn/ --fuzzy --delete-after -v
