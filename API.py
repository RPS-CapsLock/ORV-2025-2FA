from flask import Flask, request, jsonify
import numpy as np
import base64
import io
import os

def get_image():