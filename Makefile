lint:  ## Run isort, black and flake8 on app/ directory
	@echo "🔧 Sorting imports with isort..."
	@isort app/
	@echo "🧹 Removing trailing whitespace..."
	@find app/ -type f -name "*.py" -exec sed -i 's/[ \t]*$$//' {} +
	@echo "🎨 Formatting code with black..."
	@black app/
	@echo "🔍 Checking code quality with flake8..."
	@flake8 app/
	@echo "✅ Linting completed!"
