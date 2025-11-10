import { IconDefinition, IconProp } from '@fortawesome/fontawesome-svg-core';

declare module '@fortawesome/react-native-fontawesome' {
  export interface FontAwesomeIconProps {
    icon: IconDefinition | IconProp;
    size?: number;
    color?: string;
    style?: any;
  }
}
