with open('E:/Cricket-BIOMECHANICS-main1/templates/results.html', 'r', encoding='utf-8') as f:
    content = f.read()

# The exact string from the file
old_str = '<div class="card-header">\n        <span class="card-icon {% if risk.startsWith(\'High\') %}ic-red{% elif risk.startsWith(\'Moderate\') %}ic-amber{% else %}ic-green{% endif %}">\n            <i class="fas fa-shield-halved"></i>\n        </span>\n        Injury Risk & Batsman Safety\n        <span class="badge {% if risk.startsWith(\'High\') %}b-danger{% elif risk.startsWith(\'Moderate\') %}b-warning{% else %}b-success{% endif %}" style="margin-left:12px; font-size:0.7rem;">\n            {{ risk }} risk\n        </span>\n    </div>\n'

# Let's find the exact position
idx = content.find('Injury Risk')
if idx >= 0:
    start = content.rfind('<div class="card-header">', 0, idx)
    exact = content[start:start+525]  # length from check_bytes
    
    print("Exact from file:")
    print(repr(exact))
    
    # Now let's create the replacement
    # We need to modify the card-header div and add the chevron and detail div
    new_str = exact.replace(
        '<div class="card-header">',
        '<div class="card-header" onclick="toggleSection(\'injury-detail\')" style="cursor: pointer;">'
    ).replace(
        '        </span>\n    </div>',
        '        </span>\n        <i class="fas fa-chevron-down" id="injury-chevron" style="margin-left:auto; transition: transform 0.2s;"></i>\n    </div>'
    )
    
    # Add the detail div after the closing </div>
    new_str = new_str + '    <div id="injury-detail" class="section-detail" style="display: none;">'
    
    print("New string:")
    print(repr(new_str))
    
    # Replace
    content = content.replace(exact, new_str)
    
    with open('E:/Cricket-BIOMECHANICS-main1/templates/results.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS")
else:
    print("NOT FOUND")