import os
import markdown
from django.conf import settings

def load_task_description(app_folder):
    """
    Load task description from a Markdown file in the app's directory.
    
    Args:
        app_folder (str): The name of the app folder
        
    Returns:
        dict: Dictionary containing parsed sections of the task description
    """
    # Construct the path to the markdown file
    markdown_path = os.path.join(settings.BASE_DIR, app_folder, 'task_description.md')
    
    if not os.path.exists(markdown_path):
        return {
            'task_description': '<p>Task description not found.</p>',
            'learning_objectives': '<p>Learning objectives not found.</p>',
            'instructions': '<p>Instructions not found.</p>'
        }
    
    # Read the markdown file
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the markdown content into sections
    sections = parse_markdown_sections(content)
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['extra', 'codehilite'])
    
    return {
        'task_description': md.convert(sections.get('overview', '')),
        'learning_objectives': md.convert(sections.get('learning_objectives', '')),
        'instructions': md.convert(sections.get('instructions', ''))
    }

def parse_markdown_sections(content):
    """
    Parse markdown content into sections based on headers.
    
    Args:
        content (str): Raw markdown content
        
    Returns:
        dict: Dictionary with section names as keys and content as values
    """
    lines = content.split('\n')
    sections = {}
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if this is a header
        if line.startswith('## '):
            # Save previous section if it exists
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line[3:].lower().replace(' ', '_')
            current_content = []
        elif line.startswith('# '):
            # Skip the main title
            continue
        elif current_section:
            current_content.append(line)
    
    # Save the last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections 