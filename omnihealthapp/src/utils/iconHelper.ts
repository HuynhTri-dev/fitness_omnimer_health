import { IconDefinition, IconProp } from '@fortawesome/fontawesome-svg-core';

// Accept either a FontAwesome IconDefinition or a string icon name.
// If a string is given, return a FontAwesome IconProp in the form ['fas', name]
export const convertToIconProp = (icon: IconDefinition | string): IconProp => {
  if (typeof icon === 'string') {
    return ['fas', icon] as IconProp;
  }
  return ['fas', icon.iconName] as IconProp;
};
