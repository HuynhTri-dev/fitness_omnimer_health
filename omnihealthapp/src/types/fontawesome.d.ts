declare module '@fortawesome/react-native-fontawesome' {
  import { IconDefinition } from '@fortawesome/fontawesome-svg-core';

  interface FontAwesomeIconProps {
    icon: IconDefinition;
    size?: number;
    color?: string;
    style?: any;
  }

  export const FontAwesomeIcon: React.ComponentType<FontAwesomeIconProps>;
}
