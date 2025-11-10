import { library } from '@fortawesome/fontawesome-svg-core';
import {
  faEnvelope,
  faLock,
  faEye,
  faEyeSlash,
  faArrowRight,
} from '@fortawesome/free-solid-svg-icons';
import { fab } from '@fortawesome/free-brands-svg-icons';

// Add commonly used icons to the library so components can reference them by name
library.add(faEnvelope, faLock, faEye, faEyeSlash, faArrowRight, fab);

export default library;
