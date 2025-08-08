# String Value Handling in Settings Tab

## Overview
The settings tab now supports direct string editing for appropriate value types, allowing users to input custom values for strings, floats, and integers through a text input mode.

## Features

### 1. Editable Value Types
The following setting value types support string editing:
- **String**: Direct text input with no restrictions
- **Float**: Numeric input with decimal support (0.0 - 1.0 range enforced)
- **Integer**: Numeric input only (positive integers)

Non-editable types (toggle only):
- **Bool**: Space/Enter to toggle
- **Selection**: Space/Enter to cycle through options

### 2. Edit Mode Activation
- Press **Enter** on any editable setting to enter edit mode
- Visual indicator shows when in edit mode (green text color)
- Edit buffer displays current value for modification

### 3. Input Validation
Character filtering during input:
- **Float values**: Only digits and single decimal point allowed
- **Integer values**: Only numeric characters allowed
- **String values**: All characters accepted

Value validation on confirmation:
- **Float**: Must parse as valid float between 0.0 and 1.0
- **Integer**: Must parse as valid positive integer
- **String**: No validation (accepts any text)

### 4. Edit Controls
In edit mode:
- **Type characters**: Add to edit buffer (filtered by type)
- **Backspace**: Remove last character
- **Enter**: Apply changes and exit edit mode
- **Esc**: Cancel edit and restore original value

### 5. User Feedback
Toast notifications provide immediate feedback:
- ✅ Success: "Updated {setting_name}"
- ❌ Error: "Error: {validation_message}"
- ℹ️ Info: "Edit cancelled"

### 6. New String Settings
Added demonstration string settings:
- **API Endpoint**: Custom API URL (General category)
  - Default: "default" (mapped to None internally)
  - Example: "https://api.custom.com"
- **Model Name**: Default model override (Model Configuration category)
  - Default: "auto" (mapped to None internally)
  - Example: "gpt-4", "claude-2"

## Implementation Details

### SettingValue Extension
```rust
impl SettingValue {
    fn set_string(&mut self, value: String) -> Result<()> {
        match self {
            Self::String(s) => *s = value,
            Self::Float(f) => *f = value.parse()?,
            Self::Integer(i) => *i = value.parse()?,
            _ => return Err("Not editable"),
        }
        Ok(())
    }
}
```

### Edit Mode Flow
1. User presses Enter on editable setting
2. System enters edit mode, populates buffer with current value
3. User modifies value with filtered input
4. On Enter, system validates and applies changes
5. On Esc, system cancels and restores original

### Character Filtering
Prevents invalid input at the character level:
- Float: Rejects non-numeric except single decimal
- Integer: Rejects all non-numeric characters
- String: Accepts all input

## Usage Examples

### Editing Temperature (Float)
1. Navigate to "Temperature" setting
2. Press Enter to edit
3. Type "0.85" (non-numeric chars ignored)
4. Press Enter to apply
5. See "Updated Temperature" toast

### Setting Custom API Endpoint (String)
1. Navigate to "API Endpoint" setting
2. Press Enter to edit
3. Type full URL "https://api.mycompany.com/v1"
4. Press Enter to apply
5. Setting saved with custom endpoint

### Validation Errors
1. Edit Temperature, type "2.0"
2. Press Enter
3. See "Error: Invalid float value. Must be between 0.0 and 1.0"
4. Value remains unchanged

## Benefits

1. **Direct Input**: No need for incremental adjustment with arrow keys
2. **Precise Values**: Enter exact values like "0.73" for temperature
3. **Custom Strings**: Support for URLs, model names, and other text
4. **Type Safety**: Input validation prevents invalid configurations
5. **User Friendly**: Clear feedback and intuitive controls

## Future Enhancements

1. **Regex Validation**: Custom patterns for string values
2. **Auto-complete**: Suggestions for known values (model names, etc.)
3. **Multi-line Strings**: Support for longer text values
4. **Copy/Paste**: Clipboard integration for edit buffer
5. **Undo/Redo**: Edit history within the session