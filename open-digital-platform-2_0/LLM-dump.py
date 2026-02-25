import os

# Root directory of the project
root_dir = '5G_Emulator_API'
output_file = 'source-code.txt'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Walk through the directory tree
    for subdir, _, files in os.walk(root_dir):
        # Skip the virtual environment directory
        if 'venv' in subdir:
            continue
        for file in files:
            # Process only Python files
            if file.endswith('.py'):
                file_path = os.path.join(subdir, file)
                # Read the content of the Python file
                with open(file_path, 'r') as infile:
                    content = infile.read()
                # Add the location as a comment at the top
                updated_content = f"# File location: {file_path}\n{content}"
                # Write the updated content back to the file
                with open(file_path, 'w') as infile:
                    infile.write(updated_content)
                # Append the content to the output file
                outfile.write(updated_content)
                outfile.write("\n\n")

print(f"All Python files have been updated and combined into {output_file}")
