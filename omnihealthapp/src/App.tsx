import { StatusBar, StyleSheet, useColorScheme } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import LoginScreen from './presentation/screen/auth/LoginScreen';

import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

const Stack = createStackNavigator();

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <GestureHandlerRootView style={styles.rootView}>
      <SafeAreaProvider>
        <NavigationContainer>
          <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
          <Stack.Navigator>
            <Stack.Screen name="ButtonExample" component={LoginScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  rootView: {
    flex: 1,
  },
});

export default App;
