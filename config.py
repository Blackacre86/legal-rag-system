@@
-        if config_path and os.path.exists(config_path):
-            with open(config_path, 'r') as f:
-                user_config = yaml.safe_load(f)
-            self._merge_configs(self.config, user_config)
+        if config_path and os.path.exists(config_path):
+            with open(config_path, 'r') as f:
+                user_config = yaml.safe_load(f) or {}   # guard None
+            self._merge_configs(self.config, user_config)
+
+        # new: static validation
+        self.validate()
@@
     def _merge_configs(self, base: dict, update: dict) -> None:
         ...
 
+    # ───────────────────────────────────────────────────────────
+    # NEW  ►  Strong validation & default injection
+    def validate(self) -> None:
+        """
+        Ensure required keys exist and have the expected type;
+        inject any missing keys with safe defaults.
+        Raises TypeError / ValueError on incompatibilities.
+        """
+        defaults = self._load_default_config()
+
+        def _walk(default_node, current_node, path=""):
+            for key, def_val in default_node.items():
+                full = f"{path}.{key}" if path else key
+
+                # Inject missing nodes
+                if key not in current_node:
+                    current_node[key] = def_val
+
+                # Recurse into nested dictionaries
+                if isinstance(def_val, dict):
+                    if not isinstance(current_node[key], dict):
+                        raise TypeError(
+                            f"Config key '{full}' should be a mapping, "
+                            f"got {type(current_node[key]).__name__}"
+                        )
+                    _walk(def_val, current_node[key], full)
+                else:
+                    # Primitive-type validation
+                    if not isinstance(current_node[key], type(def_val)):
+                        raise TypeError(
+                            f"Config key '{full}' expects "
+                            f"{type(def_val).__name__}, "
+                            f"got {type(current_node[key]).__name__}"
+                        )
+
+        _walk(defaults, self.config)
