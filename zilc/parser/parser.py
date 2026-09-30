"""
ZIL Parser - Builds Abstract Syntax Tree from tokens.

Parses ZIL forms and creates appropriate AST nodes.
"""

from typing import List, Optional, Any
from ..lexer import Token, TokenType
from .ast_nodes import *


class Parser:
    """Parses ZIL tokens into an Abstract Syntax Tree."""

    def __init__(self, tokens: List[Token], filename: str = "<input>", mdl_zil: bool = False):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0
        # MDL-ZIL? dialect (set by <FILE-FLAGS MDL-ZIL?>). In this dialect the
        # game's routines are written with <DEFINE NAME (args) body> rather than
        # <ROUTINE ...> (bureaucracy, zorkzero), so DEFINE compiles to a Z-machine
        # routine here.  In classic ZIL, DEFINE is an MDL compile-time function /
        # macro helper (zork1's MULTIFROB), so it stays a macro definition.
        self.mdl_zil = mdl_zil
        self.current_token = self.tokens[0] if tokens else None
        # Depth of enclosing quasiquote (`) templates. While > 0, forms are DATA
        # (a template), so special-form dispatch (ROUTINE/CONSTANT/OBJECT/...) is
        # suppressed in parse_form; unquote (~) and splice-unquote (~!) escape one
        # level and restore normal dispatch. At depth 0 nothing changes.
        self.quasiquote_depth = 0

    def error(self, message: str):
        """Raise a parser error with location information."""
        if self.current_token:
            raise SyntaxError(
                f"{self.filename}:{self.current_token.line}:{self.current_token.column}: {message}"
            )
        else:
            raise SyntaxError(f"{self.filename}: {message}")

    def peek(self, offset: int = 0) -> Optional[Token]:
        """Peek at token at current position + offset."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current_token
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error."""
        if self.current_token.type != token_type:
            self.error(f"Expected {token_type.name}, got {self.current_token.type.name} ({repr(self.current_token.value)})")
        return self.advance()

    def parse(self) -> Program:
        """Parse the entire program."""
        program = Program()
        self._extra_forms = []  # For forms extracted from parenthesized groups

        while self.current_token.type != TokenType.EOF:
            node = self.parse_top_level()

            # Skip None nodes (comments, standalone strings, etc.)
            if node is None:
                continue

            # Handle lists of nodes (from VERSION? expansion)
            if isinstance(node, list):
                for n in node:
                    self._add_node_to_program(program, n)
            else:
                self._add_node_to_program(program, node)

            # Process any extra forms from parenthesized groups
            while self._extra_forms:
                extra = self._extra_forms.pop(0)
                if extra:
                    self._add_node_to_program(program, extra)

        return program

    def _add_node_to_program(self, program, node):
        """Helper to add a single node to the program."""
        if node is None:
            return

        if isinstance(node, RoutineNode):
            program.routines.append(node)
        elif isinstance(node, ObjectNode):
            program.objects.append(node)
        elif isinstance(node, RoomNode):
            program.rooms.append(node)
        elif isinstance(node, GlobalNode):
            program.globals.append(node)
        elif isinstance(node, ConstantNode):
            program.constants.append(node)
        elif isinstance(node, PropdefNode):
            program.propdefs.append(node)
        elif isinstance(node, SyntaxNode):
            program.syntax.append(node)
        elif isinstance(node, MacroNode):
            program.macros.append(node)
        elif isinstance(node, VersionNode):
            program.version = node.version
            program.version_explicit = True
        elif isinstance(node, TableNode):
            program.tables.append(node)
        elif isinstance(node, BuzzNode):
            # Add all buzz words to program's buzz_words list
            program.buzz_words.extend(node.words)
        elif isinstance(node, NewAddWordNode):
            # Add NEW-ADD-WORD declaration
            program.new_add_words.append(node)
        elif isinstance(node, SynonymNode):
            # Add all synonym words to program's synonym_words list
            program.synonym_words.extend(node.words)
            # Also track as a verb synonym group for VTBL linkage
            if len(node.words) >= 2:
                program.verb_synonym_groups.append(node.words)
        elif isinstance(node, DirectionsNode):
            # Add all direction names to program's directions list
            program.directions.extend(node.names)
        elif isinstance(node, BitSynonymNode):
            # Add bit synonym to program's bit_synonyms list
            program.bit_synonyms.append(node)
        elif isinstance(node, PrepSynonymNode):
            # Add prep synonym to program's prep_synonyms list
            program.prep_synonyms.append(node)
        elif isinstance(node, RemoveSynonymNode):
            # Add to removed synonyms list
            program.removed_synonyms.append(node.word)
        elif isinstance(node, TellTokensNode):
            # Add all tell tokens to program's tell_tokens dict
            for token_def in node.tokens:
                program.tell_tokens[token_def.name] = token_def
        elif isinstance(node, OrderObjectsNode):
            # Set object ordering mode
            program.order_objects = node.ordering
        elif isinstance(node, OrderTreeNode):
            # Set tree ordering mode
            program.order_tree = node.ordering
        elif isinstance(node, LongWordsNode):
            # Enable long word table generation
            program.long_words = True
        elif isinstance(node, DefineGlobalsNode):
            # Add DEFINE-GLOBALS declaration to program
            program.define_globals.append(node)
        elif isinstance(node, FormNode):
            # Handle top-level forms like SETG
            if isinstance(node.operator, AtomNode):
                op_name = node.operator.value.upper()
                if op_name == 'SETG' and len(node.operands) >= 2:
                    # SETG creates/updates a global variable
                    # Treat as GlobalNode for compilation
                    name_node = node.operands[0]
                    if isinstance(name_node, AtomNode):
                        global_node = GlobalNode(
                            name_node.value,
                            node.operands[1],
                            node.line,
                            node.column,
                            from_setg=True
                        )
                        program.globals.append(global_node)
                elif op_name == 'MSETG' and len(node.operands) >= 2:
                    # MSETG is ZILCH's macro-time SETG: it names a COMPILE-TIME
                    # constant (usable as ,NAME and inside table / DEFINE-GLOBALS
                    # initializers), not a runtime Z-machine global.  Route it
                    # through the CONSTANT machinery so the value folds at compile
                    # time -- values may reference earlier MSETG constants via
                    # arithmetic, e.g. <MSETG HATCH-OBJECT <+ ,OBJECT-BREAK 1>>.
                    name_node = node.operands[0]
                    if isinstance(name_node, AtomNode):
                        program.constants.append(ConstantNode(
                            name_node.value,
                            node.operands[1],
                            node.line,
                            node.column
                        ))
                elif op_name in ('ZPUT', 'PUTB', 'ZGET', 'ZREST'):
                    # Compile-time table manipulation operations
                    program.compile_time_ops.append(node)
                elif op_name == 'PUTPROP':
                    # PUTPROP atom indicator [value]
                    # Handle PROPSPEC: <PUTPROP DIRECTIONS PROPSPEC> or <PUTPROP FOO PROPSPEC FOO-HANDLER>
                    if len(node.operands) >= 2:
                        item_node = node.operands[0]
                        indicator_node = node.operands[1]
                        has_value = len(node.operands) >= 3
                        if isinstance(item_node, AtomNode) and isinstance(indicator_node, AtomNode):
                            if indicator_node.value.upper() == 'PROPSPEC':
                                if not has_value:
                                    # Clear PROPSPEC for this atom
                                    program.cleared_propspecs.add(item_node.value.upper())
                                else:
                                    # Set PROPSPEC handler for this atom
                                    value_node = node.operands[2]
                                    if isinstance(value_node, AtomNode):
                                        program.propspec_handlers[item_node.value.upper()] = value_node.value
                else:
                    # Other top-level forms (potential macro calls) - store for evaluation
                    program.top_level_forms.append(node)
            else:
                # FormNode with non-atom operator - store for evaluation
                program.top_level_forms.append(node)

    def parse_top_level(self) -> ASTNode:
        """Parse a top-level form."""
        if self.current_token.type == TokenType.LANGLE:
            return self.parse_form()
        elif self.current_token.type == TokenType.STRING:
            # Skip standalone strings (comments/documentation)
            self.advance()
            return None
        elif self.current_token.type == TokenType.ATOM and self.current_token.value in ('\\', '\\\\'):
            # Skip standalone backslash (page break marker)
            self.advance()
            return None
        elif self.current_token.type == TokenType.ATOM:
            # MDL compilation switches and other bare atoms at top level
            # Examples: ON!-INITIAL, OFF!-INITIAL
            # These are compile-time directives that we can ignore
            self.advance()
            return None
        elif self.current_token.type == TokenType.RANGLE:
            # Skip stray closing brackets (may result from complex macro expansions)
            self.advance()
            return None
        elif self.current_token.type == TokenType.RPAREN:
            # Skip stray closing parens (may result from complex macro/preprocessing)
            self.advance()
            return None
        elif self.current_token.type == TokenType.LPAREN:
            # Parenthesized forms at top level (result from %<COND> splicing)
            # These are like: (<GLOBAL NAME VALUE> <GLOBAL NAME2 VALUE2>)
            # Parse the contents as if they were top-level forms
            self.advance()  # skip (
            results = []
            while self.current_token.type != TokenType.RPAREN and self.current_token.type != TokenType.EOF:
                if self.current_token.type == TokenType.LANGLE:
                    form = self.parse_form()
                    if form:
                        results.append(form)
                else:
                    # Skip unexpected tokens inside parens
                    self.advance()
            if self.current_token.type == TokenType.RPAREN:
                self.advance()  # skip )
            # Return forms to be added to program - but parse_top_level expects single result
            # Store extra forms for later processing
            if results:
                if len(results) == 1:
                    return results[0]
                else:
                    # Multiple forms - store extras for processing
                    self._extra_forms = getattr(self, '_extra_forms', [])
                    self._extra_forms.extend(results[1:])
                    return results[0]
            return None
        elif self.current_token.type == TokenType.LBRACKET:
            # In Infocom ZIL source, [...] brackets are used as section groupings
            # but contain valid definitions that need to be parsed
            self.advance()  # Skip [
            # Parse content inside brackets as top-level forms
            results = []
            while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
                form = self.parse_top_level()
                if form is not None:
                    if isinstance(form, list):
                        results.extend(form)
                    else:
                        results.append(form)
            if self.current_token.type == TokenType.RBRACKET:
                self.advance()  # Skip ]
            # Return the first form, queue the rest
            if results:
                if len(results) > 1:
                    self._extra_forms.extend(results[1:])
                return results[0]
            return None
        elif self.current_token.type == TokenType.RBRACKET:
            # Skip stray closing bracket
            self.advance()
            return None
        elif self.current_token.type == TokenType.STRING:
            # Bare strings at top level are used as documentation in Infocom ZIL
            # They evaluate to themselves but are ignored
            self.advance()
            return None
        elif self.current_token.type == TokenType.LOCAL_VAR:
            # Local variable references at top level can occur in MDL/ZIL
            # after macro expansion or as orphaned references - skip them
            self.advance()
            return None
        elif self.current_token.type == TokenType.GLOBAL_VAR:
            # Global variable references at top level - skip them
            self.advance()
            return None
        elif self.current_token.type == TokenType.QUOTE:
            # Quoted form at top level: '< ... > or '(...)
            # This is MDL data that we can skip
            self.advance()  # skip '
            if self.current_token.type == TokenType.LANGLE:
                # Skip the quoted angle form
                depth = 1
                self.advance()  # skip <
                while depth > 0 and self.current_token.type != TokenType.EOF:
                    if self.current_token.type == TokenType.LANGLE:
                        depth += 1
                    elif self.current_token.type == TokenType.RANGLE:
                        depth -= 1
                    self.advance()
            elif self.current_token.type == TokenType.LPAREN:
                # Skip the quoted paren form
                depth = 1
                self.advance()  # skip (
                while depth > 0 and self.current_token.type != TokenType.EOF:
                    if self.current_token.type == TokenType.LPAREN:
                        depth += 1
                    elif self.current_token.type == TokenType.RPAREN:
                        depth -= 1
                    self.advance()
            return None
        else:
            self.error(f"Unexpected token at top level: {self.current_token.type.name}")

    def parse_form(self) -> ASTNode:
        """Parse a form: <operator operand1 operand2 ...>"""
        line = self.current_token.line
        col = self.current_token.column

        self.expect(TokenType.LANGLE)

        if self.current_token.type == TokenType.RANGLE:
            # Empty form <>
            self.advance()
            return FormNode(AtomNode("<>", line, col), [], line, col)

        # Parse operator
        operator = self.parse_expression()

        # Check for special forms. Inside a quasiquote template (`<...>) a form is
        # data, not a definition, so skip the special dispatch and parse it as a
        # generic form -- the operand loop below handles unquotes (~ / ~!). This is
        # what lets library macros like `<CONSTANT ~.NAME <ITABLE ...>> parse.
        if self.quasiquote_depth == 0 and isinstance(operator, AtomNode):
            op_name = operator.value.upper()

            if op_name == "ROUTINE":
                node = self.parse_routine(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "OBJECT":
                node = self.parse_object(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "ROOM":
                node = self.parse_room(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "SYNTAX":
                node = self.parse_syntax(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "VERSION":
                version_num = self.parse_expression()
                # Handle VERSION ZIP (Z-code Interpreter Program) = version 3
                if isinstance(version_num, AtomNode):
                    if version_num.value.upper() == "ZIP":
                        version_value = 3
                    elif version_num.value.upper() == "EZIP":
                        version_value = 4
                    elif version_num.value.upper() == "XZIP":
                        version_value = 5
                    elif version_num.value.upper() == "YZIP":
                        version_value = 6
                    elif version_num.value.upper() == "GLULX":
                        version_value = 256  # Glulx VM
                    else:
                        self.error(f"Unknown version name: {version_num.value}")
                elif isinstance(version_num, NumberNode):
                    version_value = version_num.value
                else:
                    self.error("VERSION requires a number or version name (ZIP/EZIP/XZIP/YZIP/GLULX)")
                # Skip any additional arguments (e.g., TIME in <VERSION ZIP TIME>)
                while self.current_token.type != TokenType.RANGLE:
                    self.advance()
                self.expect(TokenType.RANGLE)
                return VersionNode(version_value, line, col)

            elif op_name == "GLOBAL":
                node = self.parse_global(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "CONSTANT":
                node = self.parse_constant(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "PROPDEF":
                node = self.parse_propdef(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "DEFMAC":
                node = self.parse_defmac(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "DEFINE":
                # In the MDL-ZIL? dialect, <DEFINE NAME [ACT] (args) body> is the
                # game's routine-definition form (bureaucracy has no <ROUTINE> at
                # all; even GO is <DEFINE GO () ...>).  Compile it to a Z-machine
                # routine.  In classic ZIL, DEFINE names a compile-time MDL
                # function used inside macro expansion, so keep it as a macro.
                if self.mdl_zil:
                    node = self.parse_routine(line, col)
                else:
                    # DEFINE is like DEFMAC but for compile-time only
                    # Parse it the same way but mark as compile-time
                    node = self.parse_defmac(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "DEFINE20":
                # DEFINE20 is the MDL-ZIL compile-time-only function definition
                # (the "20" MDL package): form builders, MULTIFROB, etc. It is
                # NEVER a Z-machine routine -- it runs during macro expansion --
                # so always parse it as a macro definition, exactly like classic
                # DEFINE. (Its body may use MDL-only ops such as <ASSIGNED? arg>.)
                node = self.parse_defmac(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "EVAL":
                # EVAL is compile-time evaluation
                # Parse as a regular form - MDL evaluator will handle it during macro expansion
                operands = []
                while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                    operands.append(self.parse_expression())
                self.expect(TokenType.RANGLE)
                return FormNode(AtomNode(op_name, line, col), operands, line, col)

            elif op_name == "DEFAULT-DEFINITION":
                # DEFAULT-DEFINITION contains default macro definitions
                # Parse name and body expressions but don't generate code
                if self.current_token.type == TokenType.ATOM:
                    self.advance()  # Skip name
                # Parse all expressions until closing >
                while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                    self.parse_expression()
                self.expect(TokenType.RANGLE)
                return None

            elif op_name == "VERSION?":
                # VERSION? conditional compilation
                # Syntax: <VERSION? (ZIP body...) (EZIP body...) (ELSE body...)>
                # Parse each clause and only return the matching one for our version
                from .ast_nodes import Program


                # Get version from parent or default to 3
                version = 3  # Default
                # Try to get from program globals if available

                selected_bodies = []
                found_match = False

                while self.current_token.type == TokenType.LPAREN:
                    self.advance()  # Skip (

                    # Parse condition name (ZIP, EZIP, XZIP, ELSE, etc.)
                    clause_type = None
                    if self.current_token.type == TokenType.ATOM:
                        clause_type = self.current_token.value.upper()
                        self.advance()

                    # Parse body expressions for this clause
                    clause_bodies = []
                    while self.current_token.type not in (TokenType.RPAREN, TokenType.EOF):
                        clause_bodies.append(self.parse_expression())

                    self.expect(TokenType.RPAREN)

                    # Check if this clause matches our version
                    if not found_match:
                        if clause_type == "ZIP" and version == 3:
                            selected_bodies = clause_bodies
                            found_match = True
                        elif clause_type == "EZIP" and version == 4:
                            selected_bodies = clause_bodies
                            found_match = True
                        elif clause_type == "XZIP" and version == 5:
                            selected_bodies = clause_bodies
                            found_match = True
                        elif clause_type in ("ELSE", "T") and not found_match:
                            # T is MDL's true/else pattern, treat as fallback
                            selected_bodies = clause_bodies
                            found_match = True

                self.expect(TokenType.RANGLE)


                # Return the selected bodies as a list of nodes
                # These will be added to the program as if they were top-level
                return selected_bodies if selected_bodies else None

            elif op_name == "NEWTYPE":
                # NEWTYPE type-name base-type type-spec
                # Example: <NEWTYPE RSEC VECTOR '!<VECTOR ATOM ...>>
                # This is a compile-time type definition - we store it but don't generate code
                node = self.parse_newtype(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "OFFSET":
                # OFFSET index type-name field-type
                # Example: <OFFSET 1 RSEC ATOM>
                # Returns the index as a compile-time constant
                node = self.parse_offset(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name in ("PACKAGE", "ENDPACKAGE"):
                # MDL module system - skip these directives
                # Parse any arguments and discard
                while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                    self.parse_expression()
                self.expect(TokenType.RANGLE)
                return None

            elif op_name == "ENTRY":
                # ENTRY symbol1 symbol2 ... - module exports
                # Parse and discard
                while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                    self.parse_expression()
                self.expect(TokenType.RANGLE)
                return None

            elif op_name == "USE":
                # USE "module-name" - module import
                # Parse and discard
                while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                    self.parse_expression()
                self.expect(TokenType.RANGLE)
                return None

            elif op_name == "BUZZ":
                node = self.parse_buzz(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "NEW-ADD-WORD":
                node = self.parse_new_add_word(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "DIRECTIONS":
                # <DIRECTIONS NORTH SOUTH EAST WEST>
                # Parse all direction names
                names = []
                while self.current_token.type == TokenType.ATOM:
                    names.append(self.current_token.value.upper())
                    self.advance()
                self.expect(TokenType.RANGLE)
                return DirectionsNode(names, line, col)

            elif op_name == "ORDER-OBJECTS?":
                # <ORDER-OBJECTS? ROOMS-FIRST>
                # Directive for object ordering
                if self.current_token.type == TokenType.ATOM:
                    ordering = self.current_token.value.upper()
                    self.advance()
                else:
                    self.error("Expected ordering mode for ORDER-OBJECTS?")
                self.expect(TokenType.RANGLE)
                return OrderObjectsNode(ordering, line, col)

            elif op_name == "ORDER-TREE?":
                # <ORDER-TREE? REVERSE-DEFINED>
                # Directive for object tree ordering
                if self.current_token.type == TokenType.ATOM:
                    ordering = self.current_token.value.upper()
                    self.advance()
                else:
                    self.error("Expected ordering mode for ORDER-TREE?")
                self.expect(TokenType.RANGLE)
                return OrderTreeNode(ordering, line, col)

            elif op_name == "LONG-WORDS?":
                # <LONG-WORDS?>
                # Enable long word table generation
                self.expect(TokenType.RANGLE)
                return LongWordsNode(line, col)

            elif op_name == "DEFINE-GLOBALS":
                # <DEFINE-GLOBALS table-name (name val) (name BYTE val) ...>
                node = self.parse_define_globals(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name in ("SYNONYM", "VERB-SYNONYM", "ADJ-SYNONYM",
                             "DIR-SYNONYM"):
                # Standalone SYNONYM declaration (not in an object).
                # VERB-SYNONYM is the ZILCH spelling used by the
                # hollywood-hijinx-era syntax files (<VERB-SYNONYM PUSH PRESS
                # THRUST NUDGE RING>) and means exactly the same thing: the
                # aliases inherit the head word's dictionary data.  It was not
                # dispatched at all, so it fell through to a generic FormNode
                # and every alias was simply missing from the vocabulary
                # ("press" -> "this story doesn't recognize the word").
                # ADJ-SYNONYM / DIR-SYNONYM are the same family (lurkinghorror
                # declares <ADJ-SYNONYM HIGH HI>; without it "press hi" parsed
                # as an unknown usage and the microwave could not be set).
                node = self.parse_synonym_declaration(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "BIT-SYNONYM":
                # BIT-SYNONYM flag alias declaration
                node = self.parse_bit_synonym(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "PREP-SYNONYM":
                # PREP-SYNONYM preposition synonym declaration
                node = self.parse_prep_synonym(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "REMOVE-SYNONYM":
                # REMOVE-SYNONYM word declaration
                node = self.parse_remove_synonym(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "PUTPROP":
                # PUTPROP atom indicator [value]
                # With no value, clears the property
                node = self.parse_putprop(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name in ("TELL-TOKENS", "ADD-TELL-TOKENS"):
                # TELL-TOKENS / ADD-TELL-TOKENS declaration
                # ADD-TELL-TOKENS is the same as TELL-TOKENS but adds to existing tokens
                node = self.parse_tell_tokens(line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name in ("TABLE", "ITABLE", "LTABLE", "PLTABLE"):
                # PLTABLE (classic ZIL, Suspended era) = LTABLE placed in
                # PURE (static) memory: a length-prefixed table of values.
                if op_name == "PLTABLE":
                    node = self.parse_table("LTABLE", line, col)
                    if "PURE" not in node.flags:
                        node.flags.append("PURE")
                else:
                    node = self.parse_table(op_name, line, col)
                self.expect(TokenType.RANGLE)
                return node

            elif op_name == "COND":
                # Only use special COND parsing if followed by LPAREN
                # (allows quasiquote templates like `<COND ~!.CLAUSES>)
                if self.current_token.type == TokenType.LPAREN:
                    node = self.parse_cond(line, col)
                    self.expect(TokenType.RANGLE)
                    return node
                # Fall through to generic form handling

            elif op_name == "REPEAT":
                # Only use special REPEAT parsing if followed by LPAREN
                # (allows quasiquote templates like `<REPEAT ~!.STUFF>)
                if self.current_token.type == TokenType.LPAREN:
                    node = self.parse_repeat(line, col)
                    self.expect(TokenType.RANGLE)
                    return node
                # Fall through to generic form handling

        # Parse operands for generic form
        operands = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            operands.append(self.parse_expression())

        self.expect(TokenType.RANGLE)
        return FormNode(operator, operands, line, col)

    def parse_expression(self) -> ASTNode:
        """Parse a single expression (atom, number, string, form, etc.)."""
        token = self.current_token
        line = token.line
        col = token.column

        # Handle quote operator: 'EXPR becomes <QUOTE EXPR>
        if token.type == TokenType.QUOTE:
            self.advance()
            quoted_expr = self.parse_expression()
            # Create a QUOTE form
            quote_atom = AtomNode("QUOTE", line, col)
            return FormNode(quote_atom, [quoted_expr], line, col)

        # Handle quasiquote operator: `EXPR becomes QuasiquoteNode. Inside the
        # template body, special-form dispatch is suppressed (see parse_form).
        if token.type == TokenType.ATOM and token.value == '`':
            self.advance()
            self.quasiquote_depth += 1
            quoted_expr = self.parse_expression()
            self.quasiquote_depth -= 1
            return QuasiquoteNode(quoted_expr, line, col)

        # Handle unquote operator: ~EXPR becomes UnquoteNode. Unquote escapes one
        # quasiquote level, so normal special-form dispatch resumes inside it.
        if token.type == TokenType.ATOM and token.value == '~':
            self.advance()
            saved = self.quasiquote_depth
            self.quasiquote_depth = max(0, self.quasiquote_depth - 1)
            unquoted_expr = self.parse_expression()
            self.quasiquote_depth = saved
            return UnquoteNode(unquoted_expr, line, col)

        # Handle splice-unquote operator: ~!EXPR becomes SpliceUnquoteNode (also
        # escapes one quasiquote level).
        if token.type == TokenType.ATOM and token.value == '~!':
            self.advance()
            saved = self.quasiquote_depth
            self.quasiquote_depth = max(0, self.quasiquote_depth - 1)
            spliced_expr = self.parse_expression()
            self.quasiquote_depth = saved
            return SpliceUnquoteNode(spliced_expr, line, col)

        # Handle MDL splice operator: !EXPR becomes SpliceUnquoteNode
        # This is the MDL syntax for splicing (equivalent to ~! in ZILF)
        # Used in macro bodies like <FORM PROG () !<MAPF ...>>
        if token.type == TokenType.ATOM and token.value == '!':
            self.advance()
            spliced_expr = self.parse_expression()
            return SpliceUnquoteNode(spliced_expr, line, col)

        # Handle bare comma: ,EXPR becomes <GVAL EXPR> (computed global reference)
        # This is for cases like ,~<PARSE ...> where the global name is computed
        if token.type == TokenType.COMMA:
            self.advance()
            # Parse the expression that follows
            computed_expr = self.parse_expression()
            # Wrap in a GVAL form (get global value)
            gval_atom = AtomNode("GVAL", line, col)
            return FormNode(gval_atom, [computed_expr], line, col)

        # Handle bare period: .EXPR becomes <LVAL EXPR> (computed local reference)
        # This is for cases like .~.VAR where the local name is computed
        if token.type == TokenType.PERIOD:
            self.advance()
            # Parse the expression that follows
            computed_expr = self.parse_expression()
            # Wrap in a LVAL form (get local value)
            lval_atom = AtomNode("LVAL", line, col)
            return FormNode(lval_atom, [computed_expr], line, col)

        if token.type == TokenType.LANGLE:
            # Check for empty form <>
            # peek(1) looks at the NEXT token after current
            next_tok = self.peek(1)
            if next_tok and next_tok.type == TokenType.RANGLE:
                self.advance()  # Skip <
                self.advance()  # Skip >
                return FormNode(AtomNode("<>", line, col), [], line, col)
            return self.parse_form()

        elif token.type == TokenType.ATOM:
            self.advance()
            # Handle MDL #TYPE syntax: #SPLICE (x) -> <CHTYPE (x) SPLICE>
            # This is a reader shorthand for changing the type of a value
            # BUT: #BYTE and #WORD are table element markers, not type conversions
            if token.value.startswith('#') and len(token.value) > 1:
                type_name = token.value[1:].upper()  # Remove the # prefix
                # Skip #BYTE and #WORD - these are table element markers
                if type_name not in ('BYTE', 'WORD'):
                    # Check if there's a following expression to type-convert
                    if self.current_token.type in (TokenType.LPAREN, TokenType.LBRACKET,
                                                    TokenType.LANGLE, TokenType.NUMBER,
                                                    TokenType.STRING, TokenType.ATOM):
                        value_expr = self.parse_expression()
                        # Create <CHTYPE value TYPE> form
                        chtype_atom = AtomNode("CHTYPE", line, col)
                        type_atom = AtomNode(type_name, line, col)
                        return FormNode(chtype_atom, [value_expr, type_atom], line, col)
            return AtomNode(token.value, line, col)

        elif token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value, line, col)

        elif token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value, line, col)

        elif token.type == TokenType.LOCAL_VAR:
            self.advance()
            # Strip an MDL ADECL type hint on the reference (.LOCAL:FIX -> LOCAL);
            # the DECL is a compile-time type annotation with no runtime meaning.
            return LocalVarNode(token.value.split(':', 1)[0], line, col)

        elif token.type == TokenType.GLOBAL_VAR:
            self.advance()
            # Strip an MDL ADECL type hint (,HEIGHT:FIX -> HEIGHT); otherwise the
            # decorated name looks like a distinct (undeclared) global and
            # codegen tries to allocate a fresh slot for it.
            return GlobalVarNode(token.value.split(':', 1)[0], line, col)

        elif token.type == TokenType.CHAR_LOCAL_VAR:
            self.advance()
            return CharLocalVarNode(token.value, line, col)

        elif token.type == TokenType.CHAR_GLOBAL_VAR:
            self.advance()
            return CharGlobalVarNode(token.value, line, col)

        elif token.type == TokenType.LPAREN:
            # Parse list literal (parameters, etc.)
            return self.parse_list()

        elif token.type == TokenType.LBRACKET:
            # Parse vector literal [item1 item2 ...]
            return self.parse_vector()

        elif token.type == TokenType.RPAREN:
            # This shouldn't happen - probably a parse error earlier
            self.error(f"Unexpected closing parenthesis - may indicate parsing error in enclosing form")

        elif token.type == TokenType.RANGLE:
            # Stray > - skip it and return None
            # This happens with source files like Beyond Zork that have bracket imbalances
            self.advance()
            return None

        else:
            self.error(f"Unexpected token in expression: {token.type.name}")

    def parse_list(self) -> List[Any]:
        """Parse a parenthesized list."""
        self.expect(TokenType.LPAREN)
        items = []

        while self.current_token.type != TokenType.RPAREN:
            if self.current_token.type == TokenType.EOF:
                self.error("Unclosed list")

            items.append(self.parse_expression())

        self.expect(TokenType.RPAREN)
        return items

    def parse_vector(self) -> List[Any]:
        """Parse a vector literal [item1 item2 ...]."""
        line = self.current_token.line
        col = self.current_token.column
        self.expect(TokenType.LBRACKET)
        items = []

        while self.current_token.type != TokenType.RBRACKET:
            if self.current_token.type == TokenType.EOF:
                self.error("Unclosed vector")

            items.append(self.parse_expression())

        self.expect(TokenType.RBRACKET)
        # MdlVector behaves like a list everywhere but satisfies
        # <TYPE? x VECTOR> (MAP-SCOPE option dispatch relies on it).
        return MdlVector(items)

    def _normalize_param_decl(self, name: str) -> str:
        """Strip an MDL ADECL type declaration from a routine parameter name.

        Two lexed forms occur (both are compile-time type hints with no Z-machine
        representation, so the DECL is discarded and only the base name kept):

            NAME:FIX        the ADECL is glued to the atom -> return NAME
            NAME:<OR ...>   the lexer emits atom "NAME:" followed by a separate
                            <...> DECL form -> return NAME and consume the form

        bureaucracy's MDL-ZIL routines use these, e.g.
        <DEFINE DESCRIBE-SEAT (N:<OR FIX FALSE> PERS ...) ...>.
        """
        if name.endswith(':'):
            # Trailing colon: the DECL is the following <...> form (or [...] vec).
            if self.current_token.type in (TokenType.LANGLE, TokenType.LBRACKET):
                self.parse_expression()
            return name[:-1]
        if ':' in name:
            return name.split(':', 1)[0]
        return name

    def parse_routine(self, line: int, col: int) -> RoutineNode:
        """Parse ROUTINE definition."""
        # <ROUTINE name (params "AUX" aux-vars) body...>
        # or <ROUTINE name activation-name (params "AUX" aux-vars) body...>

        # Get routine name
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected routine name")
        name = self.current_token.value
        self.advance()

        # Check for optional activation name (e.g., FOO-ACT in <ROUTINE FOO FOO-ACT (...)>)
        activation_name = None
        if self.current_token.type == TokenType.ATOM:
            # This is an activation name, add it as first aux variable
            activation_name = self.current_token.value
            self.advance()

        # Parse parameter list
        params = []
        aux_vars = []
        local_defaults = {}  # Map from variable name to default value

        # If activation name was present, add it as first aux variable
        if activation_name:
            aux_vars.append(activation_name)

        opt_params = []  # Track optional params separately for MDL0417 warning
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            in_aux = False
            in_optional = False

            while self.current_token.type != TokenType.RPAREN:
                # Handle parameter modifiers: "OPTIONAL", "OPT", "AUX", "ARGS", "TUPLE", etc.
                if self.current_token.type == TokenType.STRING:
                    modifier = self.current_token.value
                    if modifier == "AUX":
                        in_aux = True
                        in_optional = False
                    elif modifier in ("OPTIONAL", "OPT"):
                        in_optional = True
                        in_aux = False
                    elif modifier == "ARGS":
                        # Variadic arguments - treat like normal params for now
                        pass
                    # Skip other modifiers like "TUPLE", "NAME", etc.
                    self.advance()
                    continue

                # Handle parameter with default value: (name default-value)
                if self.current_token.type == TokenType.LPAREN:
                    self.advance()
                    if self.current_token.type != TokenType.ATOM:
                        self.error("Expected parameter name in default value form")
                    param_name = self.current_token.value
                    self.advance()
                    param_name = self._normalize_param_decl(param_name)
                    # Parse the default value expression (optional: an ADECL-only
                    # aux binding like (NAME:FIX) may omit it)
                    default_value = None
                    if self.current_token.type != TokenType.RPAREN:
                        default_value = self.parse_expression()
                    self.expect(TokenType.RPAREN)
                    if in_aux:
                        aux_vars.append(param_name)
                    elif in_optional:
                        opt_params.append(param_name)
                        aux_vars.append(param_name)  # Also add to aux_vars for local slot
                    else:
                        params.append(param_name)
                    # Store the default value
                    if default_value is not None:
                        local_defaults[param_name] = default_value
                    continue

                # Handle simple parameter name
                if self.current_token.type == TokenType.ATOM:
                    _raw_name = self.current_token.value
                    self.advance()
                    # Normalize AFTER advancing so a trailing-colon DECL form
                    # (Y:<OR FIX FALSE>) is consumed from the correct position.
                    param_name = self._normalize_param_decl(_raw_name)
                    if in_aux:
                        aux_vars.append(param_name)
                    elif in_optional:
                        opt_params.append(param_name)
                        aux_vars.append(param_name)  # Also add to aux_vars for local slot
                    else:
                        params.append(param_name)
                else:
                    self.error("Expected parameter name")

            self.expect(TokenType.RPAREN)

        # Parse routine body
        body = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            body.append(self.parse_expression())

        return RoutineNode(name, params, aux_vars, body, line, col, local_defaults, activation_name, opt_params)

    def parse_object(self, line: int, col: int) -> ObjectNode:
        """Parse OBJECT definition."""
        # <OBJECT name (property value) (property value) ...>

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected object name")
        name = self.current_token.value
        self.advance()

        properties = self.parse_properties()

        return ObjectNode(name, properties, line, col)

    def parse_room(self, line: int, col: int) -> RoomNode:
        """Parse ROOM definition."""
        # <ROOM name (property value) (property value) ...>

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected room name")
        name = self.current_token.value
        self.advance()

        properties = self.parse_properties()

        return RoomNode(name, properties, line, col)

    def parse_properties(self) -> dict:
        """Parse object/room properties.

        Supports two syntaxes:
        - (property value...)  - standard syntax
        - <LIST property value...>  - ZILF shorthand
        """
        properties = {}
        # Track property aliases (IN and LOC are the same, etc.)
        location_props = {'IN', 'LOC'}

        while self.current_token.type in (TokenType.LPAREN, TokenType.LANGLE, TokenType.QUOTE):
            # Handle quoted properties: '(PROP value)
            if self.current_token.type == TokenType.QUOTE:
                self.advance()  # Skip quote
                # Now expect a property definition
                if self.current_token.type != TokenType.LPAREN:
                    self.error("Expected property after quote")
                # Continue to parse the property normally
                # (the quote is just for metaprogramming, we can ignore it during parsing)

            if self.current_token.type == TokenType.LANGLE:
                # Could be <LIST property value...> or a computed property like %<VERSION? ...>
                # Save position before advancing, in case we need to backtrack
                start_pos = self.pos
                self.advance()  # <

                # Check for LIST
                if self.current_token.type == TokenType.ATOM and self.current_token.value == "LIST":
                    # <LIST property value...> syntax
                    self.advance()  # LIST

                    # Get property name
                    if self.current_token.type != TokenType.ATOM:
                        self.error("Expected property name in LIST")
                    prop_name = self.current_token.value
                    self.advance()

                    # Get property value(s)
                    values = []
                    while self.current_token.type != TokenType.RANGLE:
                        if self.current_token.type == TokenType.EOF:
                            self.error("Unclosed LIST property")
                        values.append(self.parse_expression())

                    self.expect(TokenType.RANGLE)

                    # Check for duplicate property (but FLAGS can be combined)
                    # Also, IN can appear twice: (IN OBJECT) for location, (IN "string") for NEXIT
                    # Also, IN can be a direction if declared in <DIRECTIONS>, in which case
                    # (IN TO ROOM) or (IN PER ROUTINE) is a direction exit, not a location
                    is_nexit_string = (prop_name in location_props and len(values) == 1 and
                                       isinstance(values[0], StringNode))
                    # Check if this looks like a direction exit syntax: TO, PER, SORRY, NEXIT, UEXIT, etc.
                    is_direction_exit = False
                    if values and isinstance(values[0], AtomNode):
                        first_val = values[0].value.upper()
                        if first_val in ('TO', 'PER', 'SORRY', 'NEXIT', 'UEXIT', 'NE-EXIT', 'CEXIT', 'FEXIT',
                                         'DEXIT', 'DOOR', 'SETG', 'NONE', 'IF'):
                            is_direction_exit = True
                    if prop_name in properties and prop_name != 'FLAGS' and not is_nexit_string and not is_direction_exit:
                        # ZILCH allows duplicate properties - later value overwrites
                        pass  # Let it overwrite silently
                    # Handle location property conflicts (IN and LOC are the same)
                    # ZILCH allows having both - the later value wins.
                    # Only check when the value is not a NEXIT string and not a direction exit
                    if prop_name in location_props and not is_nexit_string and not is_direction_exit:
                        for loc_prop in location_props:
                            if loc_prop in properties and loc_prop != prop_name:
                                # Remove the conflicting location property - later value wins
                                del properties[loc_prop]
                    # A location keyword (IN/LOC) doubles as the IN direction. When
                    # a direction exit like (IN TO ROOM) would overwrite an existing
                    # CONTAINER under the same key, preserve the container under the
                    # other alias so the room's parent (from (IN ROOMS)) isn't lost.
                    # See the standard-syntax site: (IN "msg") is the IN
                    # direction's NEXIT message and must not consume the
                    # object's tree location.
                    if (prop_name in location_props
                            and (is_direction_exit or is_nexit_string)
                            and prop_name in properties):
                        other = 'LOC' if prop_name == 'IN' else 'IN'
                        if other not in properties:
                            properties[other] = properties[prop_name]
                    # Store property (combine FLAGS if already present)
                    if prop_name == 'FLAGS' and prop_name in properties:
                        # Combine FLAGS values
                        existing = properties[prop_name]
                        if not isinstance(existing, list):
                            existing = [existing]
                        existing.extend(values)
                        properties[prop_name] = existing
                    elif len(values) == 1:
                        properties[prop_name] = values[0]
                    else:
                        properties[prop_name] = values
                else:
                    # Not a LIST form - could be a computed property like %<VERSION? ...>
                    # Parse it as a form and store it as a computed property
                    # Reset position to reparse the form
                    self.pos = start_pos
                    self.current_token = self.tokens[self.pos]

                    # Parse the entire form as an expression
                    form_value = self.parse_expression()

                    # Store it with a generated property name
                    # In a real compiler, this would be evaluated at compile time
                    prop_name = f"__computed_prop_{len(properties)}"
                    properties[prop_name] = form_value
            else:
                # Standard (property value...) syntax
                self.advance()  # (

                # Get property name
                if self.current_token.type != TokenType.ATOM:
                    self.error("Expected property name")
                prop_name = self.current_token.value
                self.advance()

                # Get property value(s)
                values = []
                while self.current_token.type != TokenType.RPAREN:
                    if self.current_token.type == TokenType.EOF:
                        self.error("Unclosed property")
                    # Skip semicolons (ZILF separator syntax)
                    if self.current_token.type == TokenType.SEMICOLON:
                        self.advance()
                        continue
                    values.append(self.parse_expression())

                self.expect(TokenType.RPAREN)

                # Check for duplicate property (but FLAGS can be combined)
                # Also, IN can appear twice: (IN OBJECT) for location, (IN "string") for NEXIT
                # Also, IN can be a direction if declared in <DIRECTIONS>, in which case
                # (IN TO ROOM) or (IN PER ROUTINE) is a direction exit, not a location
                is_nexit_string = (prop_name in location_props and len(values) == 1 and
                                   isinstance(values[0], StringNode))
                # Check if this looks like a direction exit syntax: TO, PER, SORRY, NEXIT, UEXIT, etc.
                is_direction_exit = False
                if values and isinstance(values[0], AtomNode):
                    first_val = values[0].value.upper()
                    if first_val in ('TO', 'PER', 'SORRY', 'NEXIT', 'UEXIT', 'NE-EXIT', 'CEXIT', 'FEXIT',
                                     'DEXIT', 'DOOR', 'SETG', 'NONE', 'IF'):
                        is_direction_exit = True
                if prop_name in properties and prop_name != 'FLAGS' and not is_nexit_string and not is_direction_exit:
                    # ZILCH allows duplicate properties - later value overwrites
                    pass  # Let it overwrite silently
                # Handle location property conflicts (IN and LOC are the same)
                # ZILCH allows having both - the later value wins.
                # Only check when the value is not a NEXIT string and not a direction exit
                if prop_name in location_props and not is_nexit_string and not is_direction_exit:
                    for loc_prop in location_props:
                        if loc_prop in properties and loc_prop != prop_name:
                            # Remove the conflicting location property - later value wins
                            del properties[loc_prop]
                # A location keyword (IN/LOC) doubles as the IN direction. When a
                # direction exit like (IN TO ROOM) would overwrite an existing
                # CONTAINER value under the same key, preserve the container under
                # the other alias so the room's parent (from (IN ROOMS)) isn't lost.
                # is_nexit_string too: (IN "msg") is the IN DIRECTION's refusal
                # message, not a new location -- without this the room's
                # (IN ROOMS) parent was overwritten and the object ended up
                # parentless (cutthroats WINDING-ROAD-1 printed no room name).
                if (prop_name in location_props
                        and (is_direction_exit or is_nexit_string)
                        and prop_name in properties):
                    other = 'LOC' if prop_name == 'IN' else 'IN'
                    if other not in properties:
                        properties[other] = properties[prop_name]
                # Store property (combine FLAGS if already present)
                if prop_name == 'FLAGS' and prop_name in properties:
                    # Combine FLAGS values
                    existing = properties[prop_name]
                    if not isinstance(existing, list):
                        existing = [existing]
                    existing.extend(values)
                    properties[prop_name] = existing
                elif len(values) == 1:
                    properties[prop_name] = values[0]
                else:
                    properties[prop_name] = values

        return properties

    def parse_syntax(self, line: int, col: int) -> SyntaxNode:
        """Parse SYNTAX definition.

        Supports verb synonyms in parentheses after the verb:
        <SYNTAX TOSS (CHUCK) OBJECT AT OBJECT = V-TOSS>
        creates synonyms CHUCK -> TOSS

        Supports scope flags after OBJECT:
        <SYNTAX FOO OBJECT (HAVE) = V-FOO>
        <SYNTAX BAR OBJECT (HAVE OPTIONAL) = V-BAR>
        """
        # <SYNTAX verb [(synonym ...)] OBJECT [(flags...)] ... = V-ROUTINE>

        pattern = []
        verb_synonyms = []
        object_flags = []  # List of flag lists, one per OBJECT
        last_was_object = False

        # Parse pattern until =
        while self.current_token.type != TokenType.ATOM or self.current_token.value != "=":
            if self.current_token.type == TokenType.EOF:
                self.error("Expected = in SYNTAX")

            if self.current_token.type == TokenType.ATOM:
                atom_val = self.current_token.value.upper()
                pattern.append(self.current_token.value)
                # Track if this is OBJECT for scope flag parsing
                last_was_object = (atom_val == 'OBJECT')
                if last_was_object:
                    # Add empty flags list for this object, may be filled by following parens
                    object_flags.append([])
                self.advance()
            elif self.current_token.type == TokenType.LPAREN:
                # Check if this is verb synonyms (right after verb, before OBJECT)
                # Verb synonyms come right after the verb word and contain only atoms
                # Object specs like (FIND ACTORBIT) contain multiple atoms or come after OBJECT
                if len(pattern) == 1:
                    # This might be verb synonyms - check if it contains only atoms
                    # Peek ahead to determine if it's verb synonyms or object specs
                    self.advance()  # Skip LPAREN
                    synonyms = []
                    is_verb_synonyms = True
                    while self.current_token.type != TokenType.RPAREN:
                        if self.current_token.type == TokenType.ATOM:
                            # Check if this looks like object spec keywords
                            if self.current_token.value.upper() in ('FIND', 'HAVE', 'MANY', 'TAKE', 'HELD', 'CARRIED', 'ON-GROUND', 'IN-ROOM', 'OPTIONAL', 'STANDARD'):
                                is_verb_synonyms = False
                                break
                            synonyms.append(self.current_token.value)
                            self.advance()
                        elif self.current_token.type == TokenType.EOF:
                            self.error("Unclosed parenthesis in SYNTAX")
                        else:
                            is_verb_synonyms = False
                            break

                    if is_verb_synonyms and synonyms:
                        verb_synonyms = synonyms
                        # Skip the closing paren
                        if self.current_token.type == TokenType.RPAREN:
                            self.advance()
                    else:
                        # Not verb synonyms - need to skip rest of paren form
                        paren_depth = 1  # Already inside the paren
                        while paren_depth > 0:
                            if self.current_token.type == TokenType.LPAREN:
                                paren_depth += 1
                            elif self.current_token.type == TokenType.RPAREN:
                                paren_depth -= 1
                            elif self.current_token.type == TokenType.EOF:
                                self.error("Unclosed parenthesis in SYNTAX")
                            self.advance()
                elif last_was_object and object_flags:
                    # Scope flags after OBJECT - capture them
                    self.advance()  # Skip LPAREN
                    flags = []
                    while self.current_token.type != TokenType.RPAREN:
                        if self.current_token.type == TokenType.ATOM:
                            flags.append(self.current_token.value.upper())
                            self.advance()
                        elif self.current_token.type == TokenType.EOF:
                            self.error("Unclosed parenthesis in SYNTAX")
                        else:
                            # Skip non-atom tokens (like FIND with bit names)
                            self.advance()
                    # Merge into the last object's flags. An OBJECT may carry
                    # SEVERAL parenthesized groups -- <SYNTAX TAKE OBJECT
                    # (FIND TAKEBIT) (ON-GROUND IN-ROOM MANY) = ...> -- and the
                    # old code replaced the list and cleared last_was_object, so
                    # the second group was silently skipped: TAKE lost its scope
                    # bits and "take all" died with [You can't use multiple
                    # direct objects].
                    object_flags[-1] = object_flags[-1] + flags
                    self.advance()  # Skip RPAREN
                else:
                    # Other parenthetical forms - skip
                    paren_depth = 0
                    while True:
                        if self.current_token.type == TokenType.LPAREN:
                            paren_depth += 1
                        elif self.current_token.type == TokenType.RPAREN:
                            paren_depth -= 1
                            if paren_depth == 0:
                                self.advance()
                                break
                        elif self.current_token.type == TokenType.EOF:
                            self.error("Unclosed parenthesis in SYNTAX")
                        self.advance()
                    last_was_object = False
            else:
                self.error("Expected atom or form in SYNTAX pattern")

        # Skip =
        self.advance()

        # Get routine name(s) - can have main action and optional pre-action
        # Example: = V-PUT PRE-PUT
        # Or with action name: = V-PUT <> PUT-WITH
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected routine name after =")
        routine_parts = [self.current_token.value]
        self.advance()

        # Capture additional routines (pre-action handlers, action names)
        # Syntax can have multiple routines: = ACTION PRE1 PRE2 ...
        # Also handle empty forms <> (used for "no preaction")
        while self.current_token.type != TokenType.RANGLE:
            if self.current_token.type == TokenType.ATOM:
                routine_parts.append(self.current_token.value)
                self.advance()
            elif self.current_token.type == TokenType.LANGLE:
                # Handle form like <> (empty - no preaction) or <expr>
                depth = 0
                form_tokens = []
                while True:
                    if self.current_token.type == TokenType.LANGLE:
                        depth += 1
                        form_tokens.append('<')
                    elif self.current_token.type == TokenType.RANGLE:
                        depth -= 1
                        if depth == 0:
                            form_tokens.append('>')
                            self.advance()
                            break
                        form_tokens.append('>')
                    elif self.current_token.type == TokenType.EOF:
                        self.error("Unclosed form in SYNTAX")
                    else:
                        form_tokens.append(str(self.current_token.value) if self.current_token.value else '')
                    self.advance()
                # Represent <> as empty placeholder
                if form_tokens == ['<', '>']:
                    routine_parts.append('<>')
            else:
                break  # Unexpected token, let caller handle it

        # Join routine parts with space so compiler can split them
        routine = ' '.join(routine_parts)

        return SyntaxNode(pattern, routine, verb_synonyms, object_flags, line, col)

    def parse_global(self, line: int, col: int) -> GlobalNode:
        """Parse GLOBAL definition."""
        # <GLOBAL name initial-value>
        # ZILF format: <GLOBAL name:TYPE initial-value> - strip :TYPE annotation

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected global variable name")
        name = self.current_token.value
        # Strip ZILF type annotation (e.g., NAME:OBJECT -> NAME)
        if ':' in name:
            name = name.split(':')[0]
        self.advance()

        initial_value = None
        if self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            initial_value = self.parse_expression()

        # ZILF extended format: <GLOBAL name:TYPE initial-value <> <> type-flags>
        # Skip any additional parameters after the initial value
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            if self.current_token.type == TokenType.LANGLE:
                # Skip empty forms <> or other forms
                self.parse_expression()
            else:
                self.advance()  # Skip atoms like BYTE

        return GlobalNode(name, initial_value, line, col)

    def parse_constant(self, line: int, col: int) -> ConstantNode:
        """Parse CONSTANT definition."""
        # <CONSTANT name value>
        # Name can be an atom, or a variable reference for computed names

        if self.current_token.type == TokenType.ATOM:
            name = self.current_token.value
            self.advance()
        elif self.current_token.type in (TokenType.LOCAL_VAR, TokenType.GLOBAL_VAR):
            # Computed constant name (compile-time metaprogramming)
            # Store as special marker for now
            name = f"<computed:{self.current_token.value}>"
            self.advance()
        elif self.current_token.type == TokenType.LANGLE:
            # Computed constant name as a form: <PARSE ...>, etc.
            name_expr = self.parse_expression()
            # Store the form expression as the name
            name = f"<computed-form>"
            # Note: in a real compiler, we'd evaluate name_expr at compile time
        else:
            self.error("Expected constant name")

        value = self.parse_expression()

        return ConstantNode(name, value, line, col)

    def parse_propdef(self, line: int, col: int):
        """Parse PROPDEF property definition.

        Simple format: <PROPDEF name default-value>
        Complex format: <PROPDEF name <> (PATTERN1) (PATTERN2) ...>
        """
        from .ast_nodes import PropdefNode

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected property name")
        name = self.current_token.value
        self.advance()

        # Default value (usually a number)
        default_value = None
        patterns = []

        if self.current_token.type != TokenType.RANGLE:
            # Check for complex PROPDEF with parenthesized pattern definitions
            # Format: <PROPDEF NAME <> (PATTERN1) (PATTERN2) ...>
            if self.current_token.type == TokenType.LANGLE:
                # Could be <> (empty) or a complex form
                self.advance()  # skip <
                if self.current_token.type == TokenType.RANGLE:
                    self.advance()  # skip > - this is <> for default value
                    default_value = None
                    # Now check for pattern definitions
                    while self.current_token.type == TokenType.LPAREN:
                        pattern = self._parse_propdef_pattern()
                        if pattern:
                            patterns.append(pattern)
                else:
                    # It's something else - parse as a form expression
                    # We've already consumed <, so parse the rest of the form
                    default_value = self.parse_form(line, col)
            else:
                default_value = self.parse_expression()

        return PropdefNode(name, default_value, patterns, line, col)

    def _parse_propdef_pattern(self):
        """Parse a single PROPDEF pattern specification.

        Format: (PROP_NAME INPUT... = OUTPUT...)

        Returns a tuple (input_elements, output_elements) where:
        - input_elements: list of (type, value, var_type) tuples
        - output_elements: list of AST nodes representing output encoding
        """
        if self.current_token.type != TokenType.LPAREN:
            return None

        self.advance()  # skip (

        input_elements = []
        output_elements = []
        in_output = False

        while self.current_token.type != TokenType.RPAREN and self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.ATOM:
                atom_val = self.current_token.value
                if atom_val == '=':
                    # Switch to output parsing
                    in_output = True
                    self.advance()
                    continue

                if in_output:
                    # In output section, atoms are typically numbers or constants
                    # Try to parse as integer
                    try:
                        output_elements.append(('LENGTH', int(atom_val)))
                    except ValueError:
                        # It's a constant reference or something else
                        output_elements.append(('ATOM', atom_val))
                    self.advance()
                else:
                    # In input section
                    # Check for VAR:TYPE pattern (like FEET:FIX)
                    if ':' in atom_val:
                        parts = atom_val.split(':', 1)
                        var_name = parts[0]
                        var_type = parts[1] if len(parts) > 1 else 'FIX'
                        input_elements.append(('CAPTURE', var_name, var_type))
                    else:
                        # Literal atom
                        input_elements.append(('LITERAL', atom_val, None))
                    self.advance()

            elif self.current_token.type == TokenType.STRING:
                str_val = self.current_token.value
                if not in_output:
                    # In input section, strings are modifiers like "OPT" or "MANY"
                    input_elements.append(('MODIFIER', str_val, None))
                else:
                    # In output section, strings like "MANY" indicate repeated output
                    output_elements.append(('MODIFIER', str_val))
                self.advance()

            elif self.current_token.type == TokenType.NUMBER:
                if in_output:
                    output_elements.append(('LENGTH', self.current_token.value))
                else:
                    input_elements.append(('LITERAL', self.current_token.value, None))
                self.advance()

            elif self.current_token.type == TokenType.LANGLE:
                # Parse a form like <WORD .FEET> or <BYTE .INCHES> or <>
                self.advance()  # skip <
                if self.current_token.type == TokenType.RANGLE:
                    # Empty form <> - means auto-calculate length
                    output_elements.append(('AUTO_LENGTH', None))
                    self.advance()  # skip >
                elif self.current_token.type == TokenType.ATOM:
                    form_type = self.current_token.value
                    self.advance()
                    form_args = []
                    while self.current_token.type != TokenType.RANGLE and self.current_token.type != TokenType.EOF:
                        if self.current_token.type == TokenType.LOCAL_VAR:
                            # Local variable reference like .FEET
                            form_args.append(('VAR', self.current_token.value))
                            self.advance()
                        elif self.current_token.type == TokenType.PERIOD:
                            # Standalone period followed by atom
                            self.advance()
                            if self.current_token.type == TokenType.ATOM:
                                form_args.append(('VAR', self.current_token.value))
                                self.advance()
                        elif self.current_token.type == TokenType.ATOM:
                            form_args.append(('ATOM', self.current_token.value))
                            self.advance()
                        elif self.current_token.type == TokenType.NUMBER:
                            form_args.append(('NUMBER', self.current_token.value))
                            self.advance()
                        else:
                            self.advance()  # skip unknown tokens
                    if self.current_token.type == TokenType.RANGLE:
                        self.advance()  # skip >
                    output_elements.append(('FORM', form_type, form_args))
                else:
                    # Skip malformed form
                    while self.current_token.type != TokenType.RANGLE and self.current_token.type != TokenType.EOF:
                        self.advance()
                    if self.current_token.type == TokenType.RANGLE:
                        self.advance()

            elif self.current_token.type == TokenType.LPAREN:
                # Nested parentheses in output - constant definition like (HEIGHTSIZE 3)
                self.advance()  # skip (
                const_name = None
                const_value = None
                if self.current_token.type == TokenType.ATOM:
                    const_name = self.current_token.value
                    self.advance()
                    # The value can be a number or a form like <WORD .FEET>
                    if self.current_token.type == TokenType.NUMBER:
                        const_value = ('NUMBER', self.current_token.value)
                        self.advance()
                    elif self.current_token.type == TokenType.LANGLE:
                        # Parse form for constant value
                        self.advance()  # skip <
                        if self.current_token.type == TokenType.ATOM:
                            form_type = self.current_token.value
                            self.advance()
                            form_args = []
                            while self.current_token.type != TokenType.RANGLE and self.current_token.type != TokenType.EOF:
                                if self.current_token.type == TokenType.LOCAL_VAR:
                                    form_args.append(('VAR', self.current_token.value))
                                    self.advance()
                                elif self.current_token.type == TokenType.PERIOD:
                                    self.advance()
                                    if self.current_token.type == TokenType.ATOM:
                                        form_args.append(('VAR', self.current_token.value))
                                        self.advance()
                                elif self.current_token.type == TokenType.ATOM:
                                    form_args.append(('ATOM', self.current_token.value))
                                    self.advance()
                                else:
                                    self.advance()
                            if self.current_token.type == TokenType.RANGLE:
                                self.advance()
                            const_value = ('FORM', form_type, form_args)
                        else:
                            while self.current_token.type != TokenType.RANGLE and self.current_token.type != TokenType.EOF:
                                self.advance()
                            if self.current_token.type == TokenType.RANGLE:
                                self.advance()
                # Skip to closing paren
                while self.current_token.type != TokenType.RPAREN and self.current_token.type != TokenType.EOF:
                    self.advance()
                if self.current_token.type == TokenType.RPAREN:
                    self.advance()
                if const_name:
                    output_elements.append(('CONSTANT', const_name, const_value))

            else:
                # Skip unknown token types
                self.advance()

        if self.current_token.type == TokenType.RPAREN:
            self.advance()  # skip )

        return (input_elements, output_elements)

    def parse_defmac(self, line: int, col: int):
        """Parse DEFMAC macro definition.

        Syntax: <DEFMAC name (params...) body>

        Parameter types:
        - 'PARAM  - quoted parameter (evaluated at macro definition time)
        - PARAM   - unquoted parameter (evaluated at expansion time)
        - "TUPLE" - variable-length parameter list
        - "AUX"   - auxiliary variables

        Examples:
        <DEFMAC ENABLE ('INT) <FORM PUT .INT ,C-ENABLED? 1>>
        <DEFMAC VERB? ("TUPLE" ATMS "AUX" (O ()) (L ())) ...>
        """
        from .ast_nodes import MacroNode

        # Parse macro name
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected macro name")
        name = self.current_token.value
        self.advance()

        # Parse parameter list
        params = []
        param_defaults = {}  # name -> default value AST (OPT/AUX bindings)
        if self.current_token.type == TokenType.LPAREN:
            self.advance()  # (

            aux_mode = False
            tuple_mode = False
            optional_mode = False

            while self.current_token.type != TokenType.RPAREN:
                if self.current_token.type == TokenType.EOF:
                    self.error("Unclosed parameter list in DEFMAC")

                # Check for special keywords
                if self.current_token.type == TokenType.STRING:
                    keyword = self.current_token.value
                    if keyword == "AUX":
                        aux_mode = True
                        tuple_mode = False
                        optional_mode = False  # AUX overrides OPTIONAL
                        self.advance()
                        continue
                    elif keyword in ("OPTIONAL", "OPT"):
                        # Optional parameters - mark subsequent params as optional
                        optional_mode = True
                        self.advance()
                        continue
                    elif keyword in ("TUPLE", "ARGS"):
                        # TUPLE and ARGS both mean variadic parameters
                        tuple_mode = True
                        self.advance()
                        # Next token should be the tuple variable name
                        if self.current_token.type != TokenType.ATOM:
                            self.error(f"Expected variable name after \"{keyword}\"")
                        param_name = self.current_token.value
                        # TUPLE params: (name, is_quoted, is_tuple, is_aux, is_optional)
                        params.append((param_name, False, True, False, False))
                        self.advance()
                        continue

                # Check for quoted parameter
                is_quoted = False
                if self.current_token.type == TokenType.QUOTE:
                    # Quote prefix for parameter: 'PARAM
                    is_quoted = True
                    self.advance()

                # Parse parameter (can be simple atom or with default value)
                if self.current_token.type == TokenType.ATOM:
                    param_name = self.current_token.value
                    # Params: (name, is_quoted, is_tuple, is_aux, is_optional)
                    params.append((param_name, is_quoted, False, aux_mode, optional_mode))
                    self.advance()
                elif self.current_token.type == TokenType.LPAREN:
                    # Variable with default value: (VAR default) or ('VAR default)
                    self.advance()  # (

                    # Check for quoted parameter name inside parens
                    param_is_quoted = False
                    if self.current_token.type == TokenType.QUOTE:
                        param_is_quoted = True
                        self.advance()

                    if self.current_token.type != TokenType.ATOM:
                        self.error("Expected variable name in binding")
                    param_name = self.current_token.value
                    self.advance()

                    # Parse default value
                    default_val = self.parse_expression()
                    param_defaults[param_name] = default_val

                    # (name, is_quoted, is_tuple, is_aux, is_optional)
                    if optional_mode:
                        # A parenthesized "OPT" binding like ('O '*) is a real
                        # OPTIONAL: it consumes a caller argument when one is
                        # present and falls back to the default otherwise.
                        # It was previously marked AUX, so the caller's
                        # argument was never bound (lurkinghorror's
                        # <P? DIG GLOBAL-HOLE *> lost GLOBAL-HOLE entirely and
                        # the macro expanded to garbage).
                        params.append((param_name, param_is_quoted, False, False, True))
                    else:
                        params.append((param_name, param_is_quoted, False, True, True))
                    self.expect(TokenType.RPAREN)
                else:
                    self.error(f"Unexpected token in parameter list: {self.current_token.type}")

            self.expect(TokenType.RPAREN)

        # Parse macro body (one or more expressions until >)
        body = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            body.append(self.parse_expression())

        # If only one expression, unwrap it for backwards compatibility
        if len(body) == 1:
            body = body[0]

        node = MacroNode(name, params, body, line, col)
        node.param_defaults = param_defaults
        return node

    def parse_buzz(self, line: int, col: int):
        """Parse BUZZ noise word declaration.

        Syntax: <BUZZ word1 word2 word3 ...>

        Example: <BUZZ A AN THE IS ARE AND OF THEN>
        """
        from .ast_nodes import BuzzNode

        words = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            if self.current_token.type == TokenType.ATOM:
                word = self.current_token.value
                # Handle escape sequences like \. and \,
                if word.startswith('\\'):
                    word = word[1:]  # Remove backslash
                words.append(word)
                self.advance()
            else:
                self.error(f"Expected word in BUZZ declaration, got {self.current_token.type}")

        return BuzzNode(words, line, col)

    def parse_new_add_word(self, line: int, col: int):
        """Parse NEW-ADD-WORD vocabulary declaration (NEW-PARSER? mode).

        Syntax: <NEW-ADD-WORD name type value flags>

        Example: <NEW-ADD-WORD FOO TOBJECT <> 12345>

        Args:
            name: Word name (atom)
            type: Classification type (atom like TOBJECT, TBUZZ, etc.)
            value: Value (often <> for false)
            flags: Integer word flags
        """
        from .ast_nodes import NewAddWordNode

        # Parse word name
        if self.current_token.type != TokenType.ATOM:
            self.error(f"Expected word name in NEW-ADD-WORD, got {self.current_token.type}")
        name = self.current_token.value.upper()
        self.advance()

        # Parse word type (optional - could be TOBJECT, TBUZZ, etc.)
        word_type = None
        if self.current_token.type == TokenType.ATOM:
            word_type = self.current_token.value.upper()
            self.advance()

        # Parse value (often <> for false)
        value = None
        if self.current_token.type == TokenType.LANGLE:
            # Parse a form
            value = self.parse_form()
        elif self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.advance()
        elif self.current_token.type == TokenType.ATOM:
            value = self.current_token.value
            self.advance()

        # Parse flags (optional integer)
        flags = 0
        if self.current_token.type == TokenType.NUMBER:
            flags = self.current_token.value
            self.advance()

        return NewAddWordNode(name, word_type, value, flags, line, col)

    def parse_synonym_declaration(self, line: int, col: int):
        """Parse standalone SYNONYM declaration (not in an object).

        Syntax: <SYNONYM word1 word2 word3 ...>

        Example: <SYNONYM NORTH N FORE FORWARD F>
        """
        from .ast_nodes import SynonymNode

        words = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            if self.current_token.type == TokenType.ATOM:
                word = self.current_token.value
                # Handle escape sequences like \. and \,
                if word.startswith('\\'):
                    word = word[1:]
                words.append(word)
                self.advance()
            else:
                self.error(f"Expected word in SYNONYM declaration, got {self.current_token.type}")

        return SynonymNode(words, line, col)

    def parse_define_globals(self, line: int, col: int):
        """Parse DEFINE-GLOBALS declaration.

        Syntax: <DEFINE-GLOBALS table-name (name val) (name BYTE val) (name:ADECL val) ...>

        Example:
            <DEFINE-GLOBALS TEST-GLOBALS
                (MY-WORD 32767)           ; word-sized, initial value 32767
                (MY-BYTE BYTE 255)        ; byte-sized, initial value 255
                (HAS-ADECL:FIX 0)         ; word-sized with :FIX annotation
            >

        Creates soft globals stored in a table rather than Z-machine global variables.
        """
        from .ast_nodes import DefineGlobalsNode, DefineGlobalEntry

        # Parse table name
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected table name in DEFINE-GLOBALS")
        table_name = self.current_token.value.upper()
        self.advance()

        entries = []

        # Parse each entry: (name [BYTE] value) or (name:ADECL [BYTE] value)
        while self.current_token.type == TokenType.LPAREN:
            self.advance()  # consume (

            if self.current_token.type != TokenType.ATOM:
                self.error("Expected global name in DEFINE-GLOBALS entry")

            # Parse name, possibly with ADECL annotation (NAME:ADECL)
            name_token = self.current_token.value
            adecl = None
            if ':' in name_token:
                parts = name_token.split(':', 1)
                name = parts[0].upper()
                adecl = parts[1].upper() if len(parts) > 1 else None
            else:
                name = name_token.upper()
            self.advance()

            # Check for BYTE keyword
            is_byte = False
            if self.current_token.type == TokenType.ATOM and self.current_token.value.upper() == 'BYTE':
                is_byte = True
                self.advance()

            # Parse the initial value as a general expression. ZILCH allows any
            # GVAL here: a literal number, a compile-time constant reference
            # (,MINIMUM-BALANCE), the false/true atoms (<> / T), a string, or a
            # nested table constructor (<TABLE 0 0 ...>).  A missing value
            # (e.g. (FLIGHT-ATTENDANT-HE/SHE:STRING)) defaults to 0.  The node
            # is resolved at compile time in _process_define_globals.
            value = 0
            value_node = None
            if self.current_token.type != TokenType.RPAREN:
                value_node = self.parse_expression()
                if isinstance(value_node, NumberNode):
                    value = value_node.value

            self.expect(TokenType.RPAREN)

            entries.append(DefineGlobalEntry(name=name, value=value, is_byte=is_byte,
                                             adecl=adecl, value_node=value_node))

        return DefineGlobalsNode(table_name, entries, line, col)

    def parse_bit_synonym(self, line: int, col: int):
        """Parse BIT-SYNONYM flag alias declaration.

        Syntax variants:
        1. <BIT-SYNONYM original-flag alias-flag>
        2. <BIT-SYNONYM group-name flag1 flag2 ...>  (Infocom style)

        Example 1: <BIT-SYNONYM TOUCHBIT TOUCHEDBIT>
        This makes TOUCHEDBIT an alias for TOUCHBIT.

        Example 2: <BIT-SYNONYM EVERYBIT OPENBIT CONTBIT>
        This defines all flags under the EVERYBIT group.
        """
        from .ast_nodes import BitSynonymNode

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected flag name in BIT-SYNONYM")
        first_flag = self.current_token.value
        self.advance()

        # Collect all remaining flags
        flags = []
        while self.current_token.type == TokenType.ATOM:
            flags.append(self.current_token.value)
            self.advance()

        if not flags:
            self.error("Expected at least one alias flag in BIT-SYNONYM")

        # For now, treat first_flag as group name and others as member flags
        # Return a node for each pair (for backwards compatibility)
        # But we need to handle this in the main parse loop
        original = first_flag
        alias = flags[0] if len(flags) == 1 else flags

        return BitSynonymNode(original, alias, line, col)

    def parse_prep_synonym(self, line: int, col: int):
        """Parse PREP-SYNONYM preposition synonym declaration.

        Syntax: <PREP-SYNONYM canonical-prep synonym-prep...>

        Example: <PREP-SYNONYM TO TOWARD TOWARDS>
        This makes TOWARD and TOWARDS synonyms of TO, sharing the same prep number.
        """
        from .ast_nodes import PrepSynonymNode

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected preposition name in PREP-SYNONYM")
        canonical = self.current_token.value
        self.advance()

        # Parse one or more synonym prepositions
        synonyms = []
        while self.current_token.type == TokenType.ATOM:
            synonyms.append(self.current_token.value)
            self.advance()

        if not synonyms:
            self.error("Expected at least one synonym preposition in PREP-SYNONYM")

        return PrepSynonymNode(canonical, synonyms, line, col)

    def parse_putprop(self, line: int, col: int):
        """Parse PUTPROP directive.

        Syntax: <PUTPROP atom indicator [value]>

        Examples:
        - <PUTPROP DIRECTIONS PROPSPEC>  ; Clear PROPSPEC for DIRECTIONS
        - <PUTPROP FOO BAR 123>          ; Set FOO's BAR property to 123

        When no value is given, clears the property. We special-case the
        top-level directive shape (both operands literal atoms) for PROPSPEC
        clearing/handler registration, but PUTPROP is also an ordinary ZIL
        builtin called at runtime with arbitrary expressions for its object
        and value, e.g. <PUTPROP .NAME STATUS-LINE-SECTION .PROPERTIES> in
        the ZILF library's status.zil -- so both operands fall back to
        general expression parsing rather than requiring literal atoms.
        """
        from .ast_nodes import FormNode

        item = self.parse_expression()
        indicator = self.parse_expression()

        # Check if there's a value (if not, this is a clear operation)
        value = None
        if self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            value = self.parse_expression()

        # Return a FormNode for PUTPROP - compiler will handle it
        operands = [item, indicator]
        if value is not None:
            operands.append(value)
        return FormNode(AtomNode('PUTPROP', line, col), operands, line, col)

    def parse_tell_tokens(self, line: int, col: int):
        """Parse TELL-TOKENS declaration.

        Syntax: <TELL-TOKENS token1 [* [* ...]] <expansion1> token2 ...>

        Extended syntax:
        - (TOKEN1 TOKEN2 ...) <expansion> - Multiple tokens with same expansion
        - *:TYPE - Typed argument (e.g., *:STRING, *:FIX)

        Example:
        <TELL-TOKENS
            (CR CRLF)        <CRLF>
            DBL *            <PRINT-DBL .X>
            DBL0             <PRINT-DBL <>>
            WUTEVA *:STRING  <PRINTI .X>
            WUTEVA *:FIX     <PRINTN .X>
            GLOB             <PRINTN ,GLOB>>

        Each token definition is:
        - TOKEN (atom) or (TOKEN1 TOKEN2 ...): Name(s) of the custom token(s)
        - * or *:TYPE (zero or more): Each * indicates an argument capture
        - <EXPANSION> (form): Code to expand to, using .X, .Y, .Z, .W for args
        """
        from .ast_nodes import TellTokensNode, TellTokenDef

        tokens = []

        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            # Token name(s) can be a single atom or a group (TOKEN1 TOKEN2 ...)
            token_names = []

            if self.current_token.type == TokenType.LPAREN:
                # Group of tokens with same expansion: (CR CRLF)
                self.advance()  # Skip (
                while self.current_token.type == TokenType.ATOM:
                    token_names.append(self.current_token.value.upper())
                    self.advance()
                if self.current_token.type != TokenType.RPAREN:
                    self.error(f"Expected ) in TELL-TOKENS token group")
                self.advance()  # Skip )
            elif self.current_token.type == TokenType.ATOM:
                token_names.append(self.current_token.value.upper())
                self.advance()
            else:
                self.error(f"Expected token name in TELL-TOKENS, got {self.current_token.type}")

            # Parse argument patterns: can be:
            # - * or *:TYPE - wildcard pattern (captures argument)
            # - ,GLOBAL - specific global var pattern (no capture)
            # - <expansion> directly (no arguments)
            arg_count = 0
            arg_type = None  # Type constraint if any
            pattern = None   # Specific pattern to match (e.g., ,PRSO)

            while self.current_token.type != TokenType.LANGLE:
                if self.current_token.type == TokenType.ATOM and self.current_token.value.startswith('*'):
                    # Wildcard pattern: * or *:TYPE
                    arg_spec = self.current_token.value
                    if ':' in arg_spec:
                        # *:TYPE format - extract type
                        parts = arg_spec.split(':', 1)
                        arg_type = parts[1].upper() if len(parts) > 1 else None
                    arg_count += 1
                    self.advance()
                elif self.current_token.type == TokenType.GLOBAL_VAR:
                    # Specific global var pattern: ,PRSO, ,PRSI, etc.
                    pattern = ('GLOBAL', self.current_token.value)
                    arg_count = 0  # Pattern match consumes the slot but doesn't capture
                    self.advance()
                elif self.current_token.type == TokenType.RANGLE:
                    # End of TELL-TOKENS
                    break
                elif self.current_token.type == TokenType.EOF:
                    break
                else:
                    self.error(f"Expected expansion form in TELL-TOKENS for {token_names}, got {self.current_token.type}")

            # Expect expansion form <...>
            if self.current_token.type != TokenType.LANGLE:
                if self.current_token.type in (TokenType.RANGLE, TokenType.EOF):
                    break  # End of TELL-TOKENS
                self.error(f"Expected expansion form in TELL-TOKENS for {token_names}, got {self.current_token.type}")

            # Parse the expansion form
            expansion = self.parse_form()

            # Validate the expansion form
            self._validate_tell_token_expansion(token_names, arg_count, expansion)

            # Create token definitions for each token name
            for token_name in token_names:
                # For overloaded tokens (same name, different type), use name:type as key
                effective_name = f"{token_name}:{arg_type}" if arg_type else token_name
                tokens.append(TellTokenDef(name=effective_name, arg_count=arg_count, expansion=expansion, pattern=pattern))

        return TellTokensNode(tokens, line, col)

    def _validate_tell_token_expansion(self, token_names: List[str], arg_count: int, expansion) -> None:
        """Validate a TELL-TOKENS expansion form.

        Validates:
        1. Expansion must be a simple call (not a complex nested expression)
        2. Capture variables (.X, .Y, .Z, .W) must match arg_count
        """
        from .ast_nodes import FormNode, AtomNode, LocalVarNode

        if not isinstance(expansion, FormNode):
            return  # Non-form expansions are allowed

        # Check for complex outputs - the expansion should be a simple call
        # Complex outputs are things like <PRINTN <* 2 .X>> where the call contains
        # a nested computation. Only simple calls like <PRINT-DBL .X> are allowed.
        self._check_for_complex_tell_expansion(token_names, expansion)

        # Check for mismatched captures - the .X, .Y, .Z, .W used should match arg_count
        used_captures = self._find_capture_vars_in_expansion(expansion)
        expected_captures = ['X', 'Y', 'Z', 'W'][:arg_count]

        # Check that all used captures are within the expected range
        for var_name in used_captures:
            var_idx = {'X': 0, 'Y': 1, 'Z': 2, 'W': 3}.get(var_name.upper(), -1)
            if var_idx >= arg_count:
                self.error(
                    f"TELL-TOKENS {token_names}: capture variable .{var_name} used but only "
                    f"{arg_count} argument(s) captured"
                )

        # Check that all expected captures are used (if there are any captures)
        # Note: This is now just a warning, not an error.
        # ZILCH allows capturing arguments that aren't directly used in expansion -
        # the callee function may use implicit context (current actor, object, etc.)
        if arg_count > 0:
            expected_set = set(expected_captures)
            if not used_captures:
                # Some tokens capture args for syntax matching but use implicit context
                # (e.g., (CAO CANO) * <PRINTCA> - PRINTCA uses implicit object)
                pass  # Allow this - it's valid ZILCH behavior
            # Note: It's okay to use fewer captures than expected (e.g., only use .X when 2 are captured)
            # What's NOT okay is using captures that don't exist (checked above)

    def _check_for_complex_tell_expansion(self, token_names: List[str], form) -> None:
        """Check if a TELL expansion contains complex (nested computation) operands."""
        from .ast_nodes import FormNode, AtomNode, LocalVarNode, GlobalVarNode, NumberNode, StringNode

        if not isinstance(form, FormNode):
            return

        # Get the operator - should be a simple atom (routine name or builtin)
        operator = form.operator
        if isinstance(operator, AtomNode):
            op_name = operator.value.upper()
            # Certain builtins are allowed to have nested forms
            if op_name in ('PRINT', 'PRINTI', 'PRINTN', 'PRINTD', 'PRINTC', 'PRINTB',
                           'PRINT-DBL', 'CRLF'):
                pass  # These are simple output calls
        elif isinstance(operator, FormNode):
            # The operator itself is a form - this is definitely complex
            self.error(
                f"TELL-TOKENS {token_names}: expansion has complex operator (nested form)"
            )
            return

        # Check each operand - they should be simple values, not nested computations
        for operand in form.operands:
            if isinstance(operand, FormNode):
                # Nested form in operand - check if it's a simple reference or a computation
                nested_op = operand.operator
                # Empty form <> is okay (evaluates to FALSE)
                if nested_op is None:
                    continue
                if isinstance(nested_op, AtomNode):
                    nested_name = nested_op.value.upper()
                    # GVAL (, syntax), LVAL (. syntax) are okay - they're just variable references
                    # Empty forms <> are okay
                    if nested_name in ('GVAL', 'LVAL', 'GASSIGNED?', 'ASSIGNED?', '<>'):
                        continue
                    # Empty form check (empty operator name or '<>')
                    if not operand.operands and nested_name in ('', '<>'):
                        continue
                    # This is a nested computation like <* 2 .X> - not allowed
                    self.error(
                        f"TELL-TOKENS {token_names}: expansion contains nested computation "
                        f"<{nested_name} ...> - only simple calls are allowed"
                    )

    def _find_capture_vars_in_expansion(self, node) -> set:
        """Find all capture variable names (.X, .Y, .Z, .W) used in an expansion."""
        from .ast_nodes import FormNode, LocalVarNode

        captures = set()

        if isinstance(node, LocalVarNode):
            if node.name.upper() in ('X', 'Y', 'Z', 'W'):
                captures.add(node.name.upper())
        elif isinstance(node, FormNode):
            for operand in node.operands:
                captures.update(self._find_capture_vars_in_expansion(operand))

        return captures

    def parse_remove_synonym(self, line: int, col: int):
        """Parse REMOVE-SYNONYM declaration.

        Syntax: <REMOVE-SYNONYM word>

        Example: <REMOVE-SYNONYM GET>
        This removes GET from being a synonym so it can be used independently.
        """
        from .ast_nodes import RemoveSynonymNode

        if self.current_token.type != TokenType.ATOM:
            self.error("Expected word in REMOVE-SYNONYM")
        word = self.current_token.value
        self.advance()

        return RemoveSynonymNode(word, line, col)

    def parse_table(self, table_type: str, line: int, col: int) -> TableNode:
        """Parse TABLE/ITABLE/LTABLE definition."""
        flags = []
        size = None
        values = []
        pattern_spec = []  # PATTERN specification: list of (type_name, is_rest) tuples

        # Check for size or bare flags (ITABLE only - LTABLE doesn't use size)
        # BYTE/WORD can appear as bare atoms before size: <ITABLE BYTE 2500>
        if table_type == "ITABLE":
            # Handle bare BYTE/WORD/NONE flags before size. NONE means "no length
            # prefix" (zorkie's ITABLE default already) -- without it in this
            # whitelist, <ITABLE NONE 50> parsed NONE as a VALUE and 50 as another,
            # producing a 2-word table instead of 50 zero words. minizork declares
            # all its parser match tables that way (<GLOBAL P-PRSO <ITABLE NONE
            # 50>> etc.), so they overlapped in memory and every noun table
            # write clobbered its neighbors ("sand" ghost objects, spurious
            # SAND-F/CHALICE-F action calls, PRSI pollution).
            while self.current_token.type == TokenType.ATOM and \
                  self.current_token.value in ('BYTE', 'WORD', 'PURE', 'LENGTH', 'NONE'):
                # A bare BYTE/WORD *specifier* (old ZILCH syntax, e.g.
                # <ITABLE BYTE 100>, starcross P-INBUF) means a LENGTH-PREFIXED
                # table: byte/word 0 holds the element count, then count zero
                # elements -- equivalent to <ITABLE 100 (BYTE LENGTH)>. It
                # differs from the parenthesized (BYTE) flag, which is
                # elements-only with no prefix.
                if self.current_token.value in ('BYTE', 'WORD') and \
                        'LENGTH' not in flags and 'NONE' not in flags:
                    flags.append('LENGTH')
                flags.append(self.current_token.value)
                self.advance()
            if self.current_token.type == TokenType.NUMBER:
                size = self.current_token.value
                self.advance()
        elif table_type == "LTABLE":
            # LTABLE can have bare flags but NOT a size parameter
            while self.current_token.type == TokenType.ATOM and \
                  self.current_token.value in ('BYTE', 'WORD', 'PURE', 'LENGTH', 'NONE'):
                flags.append(self.current_token.value)
                self.advance()

        # Check for flags (BYTE), (PURE), (PATTERN (BYTE WORD)), etc.
        # Flags can be simple atoms or nested patterns like (PATTERN (BYTE [REST WORD]))
        # A QUOTEd flag list -- '(BYTE) -- comes out of compile-time splices
        # (advent: <TABLE %<VERSION? (ZIP '(BYTE)) (ELSE #SPLICE ())> ...>);
        # the quote marks the list as literal for MDL evaluation and TABLE
        # reads it as the same flag list. Without this skip it parsed as a
        # zero DATA element, shifting every entry and word-typing the table
        # (advent's ALL-TREASURES scan then read zeros -> no treasure ever
        # scored, and the whole 350-point economy was dead).
        if self.current_token.type == TokenType.QUOTE:
            nxt = self.peek(1)
            if nxt is not None and nxt.type == TokenType.LPAREN:
                self.advance()  # consume the quote; flag parsing proceeds
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            paren_depth = 1
            in_pattern = False  # Track if we're inside a PATTERN specification
            while paren_depth > 0 and self.current_token.type != TokenType.EOF:
                if self.current_token.type == TokenType.RPAREN:
                    paren_depth -= 1
                    if paren_depth == 1 and in_pattern:
                        in_pattern = False  # Exiting PATTERN spec
                    if paren_depth > 0:
                        self.advance()
                elif self.current_token.type == TokenType.LPAREN:
                    paren_depth += 1
                    self.advance()
                elif self.current_token.type == TokenType.ATOM:
                    atom_val = self.current_token.value.upper()
                    # Only collect top-level atoms as flags (BYTE, PURE, PATTERN, etc.)
                    if paren_depth == 1:
                        flags.append(self.current_token.value)
                        if atom_val == 'PATTERN':
                            in_pattern = True  # Next nested level contains pattern spec
                    elif in_pattern and paren_depth == 2:
                        # Inside PATTERN - collect type specs
                        if atom_val in ('BYTE', 'WORD'):
                            pattern_spec.append((atom_val, False))  # (type_name, is_rest)
                    self.advance()
                elif self.current_token.type == TokenType.LBRACKET:
                    # Bracketed patterns like [REST WORD]
                    self.advance()
                    rest_type = None
                    is_rest = False
                    while self.current_token.type not in (TokenType.RBRACKET, TokenType.EOF):
                        if self.current_token.type == TokenType.ATOM:
                            atom_val = self.current_token.value.upper()
                            if atom_val == 'REST':
                                is_rest = True
                            elif atom_val in ('BYTE', 'WORD'):
                                rest_type = atom_val
                        self.advance()
                    if self.current_token.type == TokenType.RBRACKET:
                        self.advance()
                    # Add the REST specification if we're in a PATTERN
                    if in_pattern and is_rest and rest_type:
                        pattern_spec.append((rest_type, True))  # (type_name, is_rest=True)
                else:
                    # Skip other tokens inside nested patterns
                    self.advance()
            if self.current_token.type == TokenType.RPAREN:
                self.advance()  # consume final )

        # Parse table values
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            values.append(self.parse_expression())

        return TableNode(table_type, flags, size, values, line, col, pattern_spec)

    def parse_cond(self, line: int, col: int) -> CondNode:
        """Parse COND statement."""
        # <COND (condition action1 action2 ...) (condition action ...) ...>

        clauses = []

        while self.current_token.type == TokenType.LPAREN:
            self.advance()  # (

            # Parse condition
            condition = self.parse_expression()
            if condition is None:
                # Skip None results (from stray brackets)
                continue

            # Parse actions
            actions = []
            while self.current_token.type not in (TokenType.RPAREN, TokenType.RANGLE, TokenType.EOF):
                expr = self.parse_expression()
                if expr is not None:  # Skip None results (from stray brackets)
                    actions.append(expr)

            if self.current_token.type == TokenType.RPAREN:
                self.advance()
            # Tolerate EOF or RANGLE instead of RPAREN for broken source
            clauses.append((condition, actions))

        return CondNode(clauses, line, col)

    def parse_repeat(self, line: int, col: int) -> 'RepeatNode':
        """Parse REPEAT statement.

        Syntax:
        <REPEAT () body...>
        <REPEAT ((var init)...) body...>
        <REPEAT ((var init)...) (condition) body...>

        Common ZIL loop patterns:
        - Infinite loop: <REPEAT () ...>
        - Loop with exit via RETURN
        - Loop with bindings for loop variables
        """
        from .ast_nodes import RepeatNode

        # Parse bindings list (required, can be empty)
        # Syntax: ((var init) (var init) ...) or ((var init) var var ...)
        bindings = []
        if self.current_token.type == TokenType.LPAREN:
            self.advance()  # (

            # Parse variable bindings - can be (var init), just var, or expressions
            # In quasiquote contexts, bindings can be unquote expressions like ~.VAR
            while self.current_token.type != TokenType.RPAREN:
                if self.current_token.type == TokenType.LPAREN:
                    # (var init) form or (unquote-expr) for quasiquote
                    self.advance()  # (

                    # Check for unquote expression like (~.VAR)
                    if self.current_token.type == TokenType.ATOM and self.current_token.value in ('~', '~!'):
                        # This is an unquote expression in the binding
                        # Parse it as an expression and use it as the binding
                        self.pos -= 1  # Back up to re-parse with (
                        self.current_token = self.tokens[self.pos - 1] if self.pos > 0 else None
                        self.advance()  # Get back to (
                        binding_expr = self.parse_expression()
                        bindings.append((binding_expr, None))  # Store expr instead of name
                    elif self.current_token.type == TokenType.ATOM:
                        # Normal (var init) form
                        var_name = self.current_token.value
                        self.advance()

                        # Parse initial value
                        init_value = self.parse_expression()

                        self.expect(TokenType.RPAREN)
                        bindings.append((var_name, init_value))
                    else:
                        # Could be a complex expression binding
                        binding_expr = self.parse_expression()
                        self.expect(TokenType.RPAREN)
                        bindings.append((binding_expr, None))
                elif self.current_token.type == TokenType.ATOM:
                    atom_val = self.current_token.value
                    # Check for unquote: ~EXPR
                    if atom_val in ('~', '~!'):
                        binding_expr = self.parse_expression()
                        bindings.append((binding_expr, None))
                    else:
                        # Plain var form (no initialization)
                        var_name = self.current_token.value
                        self.advance()
                        bindings.append((var_name, None))
                elif self.current_token.type == TokenType.LOCAL_VAR:
                    # Local variable reference in binding (.VAR) - for quasiquote expansion
                    binding_expr = self.parse_expression()
                    bindings.append((binding_expr, None))
                else:
                    self.error(f"Expected variable binding in REPEAT, got {self.current_token.type}")

            self.expect(TokenType.RPAREN)

        # Check if there's an optional condition clause
        condition = None
        if self.current_token.type == TokenType.LPAREN:
            # Peek ahead to see if this looks like a condition
            # For now, assume first ( after bindings could be condition
            # This is a simplification - full ZIL is more complex
            saved_pos = self.pos
            saved_token = self.current_token

            self.advance()  # (
            potential_condition = self.parse_expression()
            self.expect(TokenType.RPAREN)

            # If there are more expressions, treat as condition
            # Otherwise, this was the first body expression
            if self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
                condition = potential_condition
            else:
                # Restore - this was actually the body
                self.pos = saved_pos
                self.current_token = saved_token

        # Parse body
        body = []
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            body.append(self.parse_expression())

        return RepeatNode(bindings, condition, body, line, col)

    def parse_newtype(self, line: int, col: int):
        """Parse NEWTYPE type definition.

        Syntax: <NEWTYPE type-name base-type type-spec>

        Example: <NEWTYPE RSEC VECTOR '!<VECTOR ATOM <OR '* FIX> ...>>

        This is a compile-time MDL construct. We parse and store it but
        it doesn't generate runtime code. It defines a new type based on
        a primitive type (usually VECTOR) with specific field types.
        """
        from .ast_nodes import ConstantNode, NumberNode

        # Parse type name
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected type name in NEWTYPE")
        type_name = self.current_token.value
        self.advance()

        # Parse base type (usually VECTOR, LIST, etc.)
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected base type in NEWTYPE")
        base_type = self.current_token.value
        self.advance()

        # Skip the type specification - it's MDL type info we don't use
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            self.parse_expression()

        # Return as a constant that represents the type
        # Value 0 is used as a placeholder - real MDL would track type info
        return ConstantNode(type_name, NumberNode(0, line, col), line, col)

    def parse_offset(self, line: int, col: int):
        """Parse OFFSET structure field accessor.

        Syntax: <OFFSET index type-name field-type>

        Example: <SETG RSEC-RTN <OFFSET 1 RSEC ATOM>>

        Returns the index as a number - used to access fields in typed structures.
        In Z-machine terms, this is just a constant index into a table/vector.
        """
        from .ast_nodes import NumberNode

        # Parse index
        if self.current_token.type != TokenType.NUMBER:
            self.error("Expected index number in OFFSET")
        index = self.current_token.value
        self.advance()

        # Parse type name (we track but don't use for code generation)
        if self.current_token.type != TokenType.ATOM:
            self.error("Expected type name in OFFSET")
        # type_name = self.current_token.value  # Not used in Z-machine code
        self.advance()

        # Skip field type specification
        while self.current_token.type not in (TokenType.RANGLE, TokenType.EOF):
            self.parse_expression()

        # Return the index as a number node - this is what gets used at runtime
        return NumberNode(index, line, col)


def parse(tokens: List[Token], filename: str = "<input>") -> Program:
    """Convenience function to parse tokens into AST."""
    parser = Parser(tokens, filename)
    return parser.parse()
