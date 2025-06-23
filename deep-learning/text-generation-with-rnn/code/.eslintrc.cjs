module.exports = {
  extends: ["eslint:recommended", "plugin:react/recommended", "plugin:import/recommended", "plugin:jsx-a11y/recommended", "plugin:@typescript-eslint/recommended", "eslint-config-prettier", "plugin:prettier/recommended", "plugin:tailwindcss/recommended"],
  parser: "@typescript-eslint/parser",
	parserOptions: {
    project: './tsconfig.json',
    tsconfigRootDir: __dirname,
		ecmaFeatures: {
			 jsx: true
		},
		ecmaVersion:12,
		sourceType:"module"
  },
	settings: {
    react: {
      version: "detect",
    },
    "import/resolver": {
      node: {
        paths: ["src"],
        extensions: [".js", ".jsx", ".ts", ".tsx"],
      },
      alias: {
        map: [
          ["@", "./src"]
        ],
        extensions: ['.ts', '.tsx', '.js', '.jsx', '.json']
      },
      typescript: {
        alwaysTryTypes: true,
      },
    },
  },
  plugins: ["react", "@typescript-eslint", "react-hooks", "tailwindcss"],
  rules: {
		"prettier/prettier": ["error", {"endOfLine": "auto"}],
		"tailwindcss/no-custom-classname":[0],
    "jsx-a11y/click-events-have-key-events": [0],
    "jsx-a11y/no-static-element-interactions": [0],
    quotes: ["error", "double"],
    "no-use-before-define": "off",
    "@typescript-eslint/no-use-before-define": ["error"],
    "react/function-component-definition": [
      2,
      {
        namedComponents: "arrow-function",
        unnamedComponents: "arrow-function",
      },
    ],
    "react/jsx-filename-extension": [
      "warn",
      {
        extensions: [".tsx"],
      },
    ],
    // "import/extensions": [
    //   "error",
    //   "ignorePackages",
    //   {
    //     ts: "never",
    //     tsx: "never",
    //   },
    // ],
    "no-shadow": "off",
    "@typescript-eslint/no-shadow": ["error"],
    "@typescript-eslint/explicit-function-return-type": [
      "error",
      {
        allowExpressions: true,
      },
    ],
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn",
    "import/prefer-default-export": "off",
    "react/prop-types": "off",
    "react/jsx-sort-props": [
      2,
      {
        callbacksLast: true,
        shorthandFirst: false,
        shorthandLast: true,
        ignoreCase: true,
        noSortAlphabetically: false,
      },
    ],
  },
}