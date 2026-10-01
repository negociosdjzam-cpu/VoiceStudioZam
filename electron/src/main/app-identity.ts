import type { AboutPanelOptionsOptions, App, MenuItemConstructorOptions } from 'electron';

export const DESKTOP_APP_NAME = 'AUREO VOICE STUDIO';

// Keep Electron's internal name: it controls existing userData and cache paths.
const DESKTOP_STORAGE_NAME = 'VoiceStudio';

/** Set before Electron creates the native application menu. */
export function installAppIdentity(application: Pick<App, 'setName'>): void {
  application.setName(DESKTOP_STORAGE_NAME);
}

export function createMacApplicationMenuTemplate(): MenuItemConstructorOptions[] {
  return [
    {
      label: DESKTOP_APP_NAME,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { role: 'fileMenu' },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
  ];
}

interface AboutApplication {
  setAboutPanelOptions(options: AboutPanelOptionsOptions): void;
}

interface ApplicationMenuInstaller<TMenu> {
  buildFromTemplate(template: MenuItemConstructorOptions[]): TMenu;
  setApplicationMenu(menu: TMenu): void;
}

export function installMacApplicationMenu<TMenu>(
  application: AboutApplication,
  menu: ApplicationMenuInstaller<TMenu>,
  version: string,
): void {
  application.setAboutPanelOptions({
    applicationName: DESKTOP_APP_NAME,
    applicationVersion: version,
    version,
  });
  const applicationMenu = menu.buildFromTemplate(createMacApplicationMenuTemplate());
  // Electron localizes role labels itself. Replace only the old brand in the
  // native About/Hide/Quit labels, keeping their OS language and behaviour.
  const nativeMenu = applicationMenu as {
    items?: { submenu?: { items: { label: string }[] } }[];
  };
  for (const item of nativeMenu.items?.[0]?.submenu?.items ?? []) {
    item.label = item.label.replaceAll(DESKTOP_STORAGE_NAME, DESKTOP_APP_NAME);
  }
  menu.setApplicationMenu(applicationMenu);
}
