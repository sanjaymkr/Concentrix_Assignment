    
class Project_extractor:
    @classmethod
    def filter_project_entities(cls, ents):
        project_entity_list = []
        project = {}
        skills_dict = {}  # Initialize a dictionary to collect all 'Skills' values for each 'Project'

        for label, value in ents:
            if label in ['Project', 'Organization', 'Duration', 'Designation']:
                if label == 'Project':
                    if project:
                        project.setdefault('Project',None)
                        project['Skills'] = list(skills_dict.values())  # Assign collected 'Skills' to the current project
                        # Set default values as None for missing keys
                        project.setdefault('Designation', None)
                        project.setdefault('Organization', None)
                        project.setdefault('Duration', None)
                        project_entity_list.append(project)
                    project = {}  # Start a new project dictionary
                    skills_dict = {}  # Reset the skills_dict for the next project
                project[label] = value
            elif label == 'Skills':  # Process the 'Skills' label
                skills_dict[value] = value  # Add 'Skills' value to the skills_dict for the current project

        if project:
            project['Skills'] = list(skills_dict.values())  # Assign collected 'Skills' to the last project
            # Set default values as None for missing keys
            project.setdefault('Project',None)
            project.setdefault('Designation', None)
            project.setdefault('Organization', None)
            project.setdefault('Duration', None)
            project_entity_list.append(project)
        return project_entity_list

