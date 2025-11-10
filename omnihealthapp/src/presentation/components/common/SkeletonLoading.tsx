import { COLORS, RADIUS, SPACING } from '@/presentation/theme';
import React from 'react';
import { View, StyleSheet, ViewStyle, DimensionValue } from 'react-native';
import SkeletonPlaceholder from 'react-native-skeleton-placeholder';

export enum SkeletonVariant {
  Card = 'Card',
  TextField = 'TextField',
  CircleImage = 'CircleImage',
  RectangleImage = 'RectangleImage',
  Avatar = 'Avatar',
  Line = 'Line',
  Button = 'Button',
  ListItem = 'ListItem',
}

interface SkeletonLoadingProps {
  variant: SkeletonVariant;
  width?: number | string;
  height?: number | string;
  borderRadius?: number;
  style?: ViewStyle;
  count?: number;
}

export const SkeletonLoading: React.FC<SkeletonLoadingProps> = ({
  variant,
  width = '100%',
  height = 100,
  borderRadius = RADIUS.md,
  style,
  count = 1,
}) => {
  const renderSkeleton = () => {
    switch (variant) {
      case SkeletonVariant.Card:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.card,
                {
                  width: width as DimensionValue,
                  height: height as DimensionValue,
                  borderRadius,
                },
                style,
              ]}
            >
              <View style={styles.cardImage} />
              <View style={styles.cardContent}>
                <View style={styles.cardTitle} />
                <View style={styles.cardSubtitle} />
                <View style={styles.cardFooter}>
                  <View style={styles.cardButton} />
                </View>
              </View>
            </View>
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.TextField:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.textField,
                {
                  width: width as DimensionValue,
                  height: (height || 48) as DimensionValue,
                  borderRadius: RADIUS.sm,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.CircleImage:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.circleImage,
                {
                  width: (width || 80) as DimensionValue,
                  height: (height || 80) as DimensionValue,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.RectangleImage:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.rectangleImage,
                {
                  width: width as DimensionValue,
                  height: height as DimensionValue,
                  borderRadius,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.Avatar:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View style={[styles.avatarContainer, style]}>
              <View style={styles.avatar} />
              <View style={styles.avatarText}>
                <View style={styles.avatarName} />
                <View style={styles.avatarEmail} />
              </View>
            </View>
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.Line:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.line,
                {
                  width: width as DimensionValue,
                  height: (height || 16) as DimensionValue,
                  borderRadius: RADIUS.sm,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.Button:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                styles.button,
                {
                  width: (width || 120) as DimensionValue,
                  height: (height || 40) as DimensionValue,
                  borderRadius: RADIUS.sm,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );

      case SkeletonVariant.ListItem:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View style={[styles.listItem, style]}>
              <View style={styles.listItemAvatar} />
              <View style={styles.listItemContent}>
                <View style={styles.listItemTitle} />
                <View style={styles.listItemSubtitle} />
              </View>
            </View>
          </SkeletonPlaceholder>
        );

      default:
        return (
          <SkeletonPlaceholder
            backgroundColor={COLORS.GRAY_200}
            highlightColor={COLORS.GRAY_100}
            speed={1200}
          >
            <View
              style={[
                {
                  width: width as DimensionValue,
                  height: height as DimensionValue,
                  borderRadius,
                },
                style,
              ]}
            />
          </SkeletonPlaceholder>
        );
    }
  };

  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <View key={index} style={index > 0 && styles.skeletonSpacing}>
          {renderSkeleton()}
        </View>
      ))}
    </>
  );
};

const styles = StyleSheet.create({
  // Card Skeleton
  card: {
    overflow: 'hidden',
  },
  cardImage: {
    width: '100%',
    height: 150,
  },
  cardContent: {
    padding: SPACING.md,
  },
  cardTitle: {
    width: '70%',
    height: 20,
    borderRadius: RADIUS.sm,
    marginBottom: SPACING.sm,
  },
  cardSubtitle: {
    width: '90%',
    height: 14,
    borderRadius: RADIUS.sm,
    marginBottom: SPACING.md,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: SPACING.sm,
  },
  cardButton: {
    width: 80,
    height: 32,
    borderRadius: RADIUS.sm,
  },

  // TextField Skeleton
  textField: {},

  // Circle Image Skeleton
  circleImage: {
    borderRadius: 9999,
  },

  // Rectangle Image Skeleton
  rectangleImage: {},

  // Avatar Skeleton
  avatarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
  },
  avatarText: {
    marginLeft: SPACING.md,
    flex: 1,
  },
  avatarName: {
    width: '60%',
    height: 16,
    borderRadius: RADIUS.sm,
    marginBottom: SPACING.xs,
  },
  avatarEmail: {
    width: '80%',
    height: 12,
    borderRadius: RADIUS.sm,
  },

  // Line Skeleton
  line: {},

  // Button Skeleton
  button: {},

  // List Item Skeleton
  listItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: SPACING.sm,
  },
  listItemAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
  },
  listItemContent: {
    marginLeft: SPACING.md,
    flex: 1,
  },
  listItemTitle: {
    width: '70%',
    height: 16,
    borderRadius: RADIUS.sm,
    marginBottom: SPACING.xs,
  },
  listItemSubtitle: {
    width: '50%',
    height: 12,
    borderRadius: RADIUS.sm,
  },

  // Spacing between skeletons
  skeletonSpacing: {
    marginTop: SPACING.md,
  },
});
