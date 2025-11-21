"""
Tests for code parsers.
"""

import pytest
from src.ingestion.parsers import (
    PythonParser,
    JavaScriptParser,
    CodeParser,
    CodeEntity,
    EntityType,
    ParseError
)


class TestPythonParser:
    """Test suite for PythonParser."""
    
    @pytest.fixture
    def parser(self):
        """Create a PythonParser instance."""
        return PythonParser()
    
    def test_parser_language(self, parser):
        """Test parser language property."""
        assert parser.get_language() == "python"
        assert parser.supports_language("python")
        assert not parser.supports_language("javascript")
    
    def test_parse_function(self, parser):
        """Test parsing a simple function."""
        code = """
def hello_world():
    '''Say hello to the world.'''
    print("Hello, World!")
"""
        
        entities = parser.parse(code)
        
        assert len(entities) == 1
        assert entities[0].name == "hello_world"
        assert entities[0].entity_type == EntityType.FUNCTION
        assert entities[0].docstring == "Say hello to the world."
        assert "print" in entities[0].content
    
    def test_parse_class(self, parser):
        """Test parsing a class."""
        code = """
class Calculator:
    '''A simple calculator.'''
    
    def add(self, a, b):
        '''Add two numbers.'''
        return a + b
    
    def subtract(self, a, b):
        return a - b
"""
        
        entities = parser.parse(code)
        
        # Should find the class
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Calculator"
        assert classes[0].docstring == "A simple calculator."
        assert "add" in classes[0].metadata["methods"]
        assert "subtract" in classes[0].metadata["methods"]
    
    def test_parse_with_decorators(self, parser):
        """Test parsing functions with decorators."""
        code = """
@property
def value(self):
    return self._value

@staticmethod
def create():
    return MyClass()
"""

        entities = parser.parse(code)

        assert len(entities) >= 2
        # Check that decorators are captured (without @ symbol)
        for entity in entities:
            if entity.name == "value":
                assert "property" in entity.decorators
            elif entity.name == "create":
                assert "staticmethod" in entity.decorators
    
    def test_extract_imports(self, parser):
        """Test extracting import statements."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict
"""
        
        imports = parser.extract_imports(code)
        
        assert len(imports) == 4
        assert "import os" in imports
        assert "from pathlib import Path" in imports
    
    def test_parse_invalid_syntax(self, parser):
        """Test parsing code with syntax errors."""
        code = "def broken(:\n    pass"
        
        with pytest.raises(ParseError):
            parser.parse(code)


class TestJavaScriptParser:
    """Test suite for JavaScriptParser."""
    
    @pytest.fixture
    def parser(self):
        """Create a JavaScriptParser instance."""
        return JavaScriptParser()
    
    def test_parser_language(self, parser):
        """Test parser language property."""
        assert parser.get_language() == "javascript"
        assert parser.supports_language("javascript")
        assert not parser.supports_language("python")
    
    def test_parse_function(self, parser):
        """Test parsing a function declaration."""
        code = """
function greet(name) {
    return `Hello, ${name}!`;
}
"""
        
        entities = parser.parse(code)
        
        assert len(entities) >= 1
        func = [e for e in entities if e.name == "greet"][0]
        assert func.entity_type == EntityType.FUNCTION
        assert "name" in func.parameters
    
    def test_parse_arrow_function(self, parser):
        """Test parsing arrow functions."""
        code = """
const add = (a, b) => {
    return a + b;
};

const multiply = (x, y) => x * y;
"""
        
        entities = parser.parse(code)
        
        assert len(entities) >= 2
        names = [e.name for e in entities]
        assert "add" in names
        assert "multiply" in names
    
    def test_parse_class(self, parser):
        """Test parsing a class."""
        code = """
class Counter {
    constructor() {
        this.count = 0;
    }
    
    increment() {
        this.count++;
    }
}
"""
        
        entities = parser.parse(code)
        
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 1
        assert classes[0].name == "Counter"
    
    def test_extract_imports(self, parser):
        """Test extracting import statements."""
        code = """
import React from 'react';
import { useState, useEffect } from 'react';
const fs = require('fs');
"""
        
        imports = parser.extract_imports(code)
        
        assert len(imports) >= 2
        assert any("React" in imp for imp in imports)

